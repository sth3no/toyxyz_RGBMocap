"""Full-resolution GPU SMPL-X LBS and smooth normals; no Torch in playback.

OpenGL 4.3 runs the kernels as compute shaders. OpenGL 4.1 Core (macOS) has no
compute shaders or storage buffers, so the same arithmetic runs as vertex
shaders over gl_VertexID into transform-feedback buffers and reads its inputs
through texture buffers. RGBMOCAP_GL_BACKEND=feedback forces that path on a
4.3 driver for parity checks.
"""
import ctypes
import os
import sys
import numpy as np
from shiboken6 import VoidPtr
from PySide6.QtGui import QMatrix4x4, QOpenGLContext, QVector3D
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram,
    QOpenGLFunctions_4_1_Core, QOpenGLFunctions_4_3_Core, QOpenGLTexture, QOpenGLVertexArrayObject)

SKIN = """#version 430 core
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Model { float model[]; };
struct Vertex { vec4 position; vec4 normal; };
layout(std430,binding=1) buffer Output { Vertex vertices[]; };
layout(std430,binding=4) readonly buffer Frame { float pose[]; };
uniform int vertexCount;
void main() {
    uint i=gl_GlobalInvocationID.x;
    if (i>=uint(vertexCount)) return;
    int n=vertexCount*3;
    vec3 p=vec3(model[3*i],model[3*i+1],model[3*i+2]);
    for (int k=0;k<20;k++) {
        int o=n+k*n+int(3*i);
        p+=pose[1366+k]*vec3(model[o],model[o+1],model[o+2]);
    }
    for (int k=0;k<486;k++) {
        int o=n*21+k*n+int(3*i);
        p+=pose[880+k]*vec3(model[o],model[o+1],model[o+2]);
    }
    vec4 v=vec4(p,1);
    vec3 result=vec3(0);
    for (int j=0;j<55;j++) {
        float w=model[n*507+int(i)*55+j];
        if (w==0) continue;
        int o=j*16;
        result+=w*vec3(dot(vec4(pose[o],pose[o+1],pose[o+2],pose[o+3]),v),
                       dot(vec4(pose[o+4],pose[o+5],pose[o+6],pose[o+7]),v),
                       dot(vec4(pose[o+8],pose[o+9],pose[o+10],pose[o+11]),v));
    }
    vertices[i].position=vec4(result+vec3(pose[1386],pose[1387],pose[1388]),1);
}
"""
NORMALS = """#version 430 core
layout(local_size_x=64) in;
struct Vertex { vec4 position; vec4 normal; };
layout(std430,binding=1) buffer Output { Vertex vertices[]; };
layout(std430,binding=2) readonly buffer Offsets { uint offsets[]; };
layout(std430,binding=3) readonly buffer Neighbors { uint neighbors[]; };
uniform int vertexCount;
void main() {
    uint i=gl_GlobalInvocationID.x;
    if (i>=uint(vertexCount)) return;
    vec3 center=vertices[i].position.xyz;
    vec3 n=vec3(0);
    for (uint a=offsets[i];a<offsets[i+1];a++) {
        vec3 b=vertices[neighbors[2*a]].position.xyz;
        vec3 c=vertices[neighbors[2*a+1]].position.xyz;
        n+=cross(b-center,c-center);
    }
    vertices[i].normal=vec4(normalize(n),0);
}
"""
VERTEX = """#version 430 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
uniform mat4 mvp;
uniform mat4 sceneTransform;
out vec3 surfaceNormal;
void main() { gl_Position=mvp*sceneTransform*vec4(position,1); surfaceNormal=mat3(sceneTransform)*normal; }
"""
FRAGMENT = """#version 430 core
in vec3 surfaceNormal;
out vec4 color;
uniform vec3 bodyColor;
void main() {
    vec3 n=normalize(surfaceNormal);
    float key=max(0,dot(n,normalize(vec3(-.4,.8,-.6))));
    float fill=max(0,dot(n,normalize(vec3(.6,.3,.7))));
    color=vec4(bodyColor*(.40+.55*key+.20*fill),1);
}
"""
# Transform-feedback counterparts of SKIN and NORMALS (OpenGL 4.1). A texel
# index is the storage-buffer float offset above divided by the texel width
# (RGB32F: 3, RGBA32F: 4). Positions and normals are separate vec4 outputs:
# the normals pass reads neighbouring positions and cannot write that buffer.
FEEDBACK_SKIN = """#version 410 core
uniform samplerBuffer modelVectors;
uniform samplerBuffer modelScalars;
uniform samplerBuffer poseRows;
uniform samplerBuffer poseScalars;
uniform int vertexCount;
out vec4 position;
void main() {
    int i=gl_VertexID;
    int c=vertexCount;
    vec3 p=texelFetch(modelVectors,i).xyz;
    for (int k=0;k<20;k++)
        p+=texelFetch(poseScalars,1366+k).r*texelFetch(modelVectors,(1+k)*c+i).xyz;
    for (int k=0;k<486;k++)
        p+=texelFetch(poseScalars,880+k).r*texelFetch(modelVectors,(21+k)*c+i).xyz;
    vec4 v=vec4(p,1);
    vec3 result=vec3(0);
    for (int j=0;j<55;j++) {
        float w=texelFetch(modelScalars,c*3*507+i*55+j).r;
        if (w==0) continue;
        result+=w*vec3(dot(texelFetch(poseRows,4*j),v),
                       dot(texelFetch(poseRows,4*j+1),v),
                       dot(texelFetch(poseRows,4*j+2),v));
    }
    position=vec4(result+vec3(texelFetch(poseScalars,1386).r,texelFetch(poseScalars,1387).r,
                              texelFetch(poseScalars,1388).r),1);
}
"""
FEEDBACK_NORMALS = """#version 410 core
uniform samplerBuffer positions;
uniform usamplerBuffer offsets;
uniform usamplerBuffer neighbors;
out vec4 normal;
void main() {
    int i=gl_VertexID;
    vec3 center=texelFetch(positions,i).xyz;
    vec3 n=vec3(0);
    for (uint a=texelFetch(offsets,i).r;a<texelFetch(offsets,i+1).r;a++) {
        vec3 b=texelFetch(positions,int(texelFetch(neighbors,int(2u*a)).r)).xyz;
        vec3 c=texelFetch(positions,int(texelFetch(neighbors,int(2u*a+1u)).r)).xyz;
        n+=cross(b-center,c-center);
    }
    normal=vec4(normalize(n),0);
}
"""

GL_FLOAT = 0x1406
GL_TEXTURE0 = 0x84C0
GL_TEXTURE_BUFFER = 0x8C2A
GL_MAX_TEXTURE_BUFFER_SIZE = 0x8C2B
GL_RASTERIZER_DISCARD = 0x8C89
GL_SEPARATE_ATTRIBS = 0x8C8D
GL_TRANSFORM_FEEDBACK_BUFFER = 0x8C8E
GL_POINTS = 0x0000
R32F, RGB32F, RGBA32F, R32UI = 0x822E, 0x8815, 0x8814, 0x8236
TEXEL_BYTES = {R32F: 4, RGB32F: 12, RGBA32F: 16, R32UI: 4}
BACKEND_ENV = "RGBMOCAP_GL_BACKEND"


def feedback_forced():
    return os.environ.get(BACKEND_ENV, "").strip().lower() == "feedback"


def open_functions():
    """OpenGL functions and skinning backend for the current context.

    The 4.3 compute path is used whenever the driver provides it, unless the
    transform-feedback path is forced for parity testing.
    """
    if not feedback_forced():
        gl = QOpenGLFunctions_4_3_Core()
        if gl.initializeOpenGLFunctions():
            return gl, "compute"
    gl = QOpenGLFunctions_4_1_Core()
    if gl.initializeOpenGLFunctions():
        return gl, "feedback"
    raise RuntimeError("SMPL-X Mesh requires OpenGL 4.3, or OpenGL 4.1 Core on macOS. Update the graphics driver.")


def set_feedback_varyings(program, names):
    # PySide6 has no binding for glTransformFeedbackVaryings. It must be called
    # after the shaders are attached and before linking.
    context = QOpenGLContext.currentContext()
    address = context.getProcAddress(b"glTransformFeedbackVaryings") if context is not None else None
    if not address:
        raise RuntimeError("OpenGL transform feedback is unavailable")
    factory = ctypes.WINFUNCTYPE if sys.platform == "win32" else ctypes.CFUNCTYPE
    function = factory(None, ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.c_uint)(int(address))
    values = (ctypes.c_char_p * len(names))(*[name.encode() for name in names])
    function(program.programId(), len(names), values, GL_SEPARATE_ATTRIBS)


class MeshRenderer:
    def __init__(self,shared=None):
        self.gl, self.backend = open_functions()
        self.feedback = self.backend == "feedback"
        self.shared=shared
        self.skin = shared.skin if shared else self.kernel(SKIN,FEEDBACK_SKIN)
        self.normals = shared.normals if shared else self.kernel(NORMALS,FEEDBACK_NORMALS,("normal",))
        self.surface = shared.surface if shared else self.program(((QOpenGLShader.Vertex,self.surface_source(VERTEX)),
                                                                   (QOpenGLShader.Fragment,self.surface_source(FRAGMENT))))
        self.buffers = []
        self.views = []
        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()
        self.feedback_vao = None
        if self.feedback:
            # Attribute-less VAO for the kernels; it never reads the outputs.
            self.feedback_vao = QOpenGLVertexArrayObject()
            self.feedback_vao.create()
        self.asset_key = None
        self.frame_key = None
        self.vertex_count = self.index_count = self.gpu_bytes = 0
        self.dispatches = 0
        self.driver = self.gl.glGetString(0x1F01)

    def check(self,label):
        error = self.gl.glGetError()
        if error:
            raise RuntimeError(f"OpenGL {error} at {label}")

    @staticmethod
    def program(stages,varyings=None):
        p = QOpenGLShaderProgram()
        for stage,source in stages:
            if not p.addShaderFromSourceCode(stage,source):
                raise RuntimeError(p.log())
        if varyings:
            set_feedback_varyings(p,varyings)
        if not p.link():
            raise RuntimeError(p.log())
        return p

    def kernel(self,compute,feedback,varyings=("position",)):
        if self.feedback:
            return self.program(((QOpenGLShader.Vertex,feedback),),varyings)
        return self.program(((QOpenGLShader.Compute,compute),))

    def surface_source(self,source):
        # Draw shaders use nothing beyond GLSL 4.10; only the declaration differs.
        return source.replace("#version 430 core","#version 410 core",1) if self.feedback else source

    def buffer(self, data, kind=None):
        b = QOpenGLBuffer(kind or QOpenGLBuffer.VertexBuffer)
        if not b.create() or not b.bind():
            raise RuntimeError("Cannot create a mesh GPU buffer")
        if isinstance(data,int):
            b.allocate(data)
            size = data
        else:
            value = data.tobytes()
            b.allocate(value,len(value))
            size = len(value)
        self.buffers.append(b)
        self.gpu_bytes += size
        return b

    def view(self,buffer,internal_format):
        """Texture-buffer view of a GPU buffer for the transform-feedback kernels."""
        texels = buffer.size()//TEXEL_BYTES[internal_format]
        limit = int(self.gl.glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE))
        if texels > limit:
            raise RuntimeError(f"Mesh data exceeds this GPU's texture buffer limit ({texels} > {limit} texels)")
        texture = QOpenGLTexture(QOpenGLTexture.TargetBuffer)
        if not texture.create():
            raise RuntimeError("Cannot create a mesh texture buffer")
        self.views.append(texture)
        texture.bind()
        self.gl.glTexBuffer(GL_TEXTURE_BUFFER,internal_format,buffer.bufferId())
        texture.release()
        return texture

    def feedback_outputs(self):
        self.positions = self.buffer(self.vertex_count*16)
        self.normal_buffer = self.buffer(self.vertex_count*16)

    def feedback_attributes(self,program):
        for location,buffer in ((0,self.positions),(1,self.normal_buffer)):
            buffer.bind()
            program.enableAttributeArray(location)
            program.setAttributeBuffer(location,GL_FLOAT,0,3,16)

    def normal_views(self):
        return [("positions",self.view(self.positions,RGBA32F)),("offsets",self.view(self.offsets,R32UI)),
                ("neighbors",self.view(self.neighbors,R32UI))]

    def skin_feedback_inputs(self):
        return self.skin_inputs

    def run_feedback(self,passes,uniforms=()):
        """Run kernels as point draws with rasterization off, then restore GL state."""
        gl = self.gl
        self.feedback_vao.bind()
        gl.glEnable(GL_RASTERIZER_DISCARD)
        try:
            for program,inputs,outputs in passes:
                program.bind()
                try:
                    for unit,(name,texture) in enumerate(inputs):
                        gl.glActiveTexture(GL_TEXTURE0+unit)
                        texture.bind()
                        gl.glUniform1i(program.uniformLocation(name),unit)
                    gl.glUniform1i(program.uniformLocation("vertexCount"),self.vertex_count)
                    for name,value in uniforms:
                        gl.glUniform1i(program.uniformLocation(name),value)
                    for index,buffer in enumerate(outputs):
                        gl.glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER,index,buffer.bufferId())
                    gl.glBeginTransformFeedback(GL_POINTS)
                    gl.glDrawArrays(GL_POINTS,0,self.vertex_count)
                    gl.glEndTransformFeedback()
                    self.check("transform feedback")
                finally:
                    for index in range(len(outputs)):
                        gl.glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER,index,0)
                    for unit,(_,texture) in reversed(list(enumerate(inputs))):
                        gl.glActiveTexture(GL_TEXTURE0+unit)
                        texture.release()
                    program.release()
        finally:
            gl.glDisable(GL_RASTERIZER_DISCARD)
            gl.glActiveTexture(GL_TEXTURE0)
            self.feedback_vao.release()

    def load(self, asset):
        if asset["key"] == self.asset_key:
            return
        self.release_buffers()
        self.vertex_count = int(asset["vertex_count"])
        self.index_count = asset["faces"].size
        self.vao.bind()
        owner = self.shared if self.shared and self.shared.asset_key==asset['key'] else None
        self.model = owner.model if owner else self.buffer(asset["packed"])
        if self.feedback:
            self.feedback_outputs()
        else:
            self.vertices = self.buffer(self.vertex_count*32)
        self.offsets = owner.offsets if owner else self.buffer(asset["offsets"])
        self.neighbors = owner.neighbors if owner else self.buffer(asset["neighbors"])
        self.frame = self.buffer(1389*4)
        self.indices = owner.indices if owner else self.buffer(asset["faces"],QOpenGLBuffer.IndexBuffer)
        self.indices.bind()
        if self.feedback:
            self.surface.bind()
            self.feedback_attributes(self.surface)
            self.vao.release()
            self.skin_inputs = [("modelVectors",self.view(self.model,RGB32F)),("modelScalars",self.view(self.model,R32F)),
                                ("poseRows",self.view(self.frame,RGBA32F)),("poseScalars",self.view(self.frame,R32F))]
            self.normal_inputs = self.normal_views()
        else:
            for index,b in enumerate((self.model,self.vertices,self.offsets,self.neighbors,self.frame)):
                self.gl.glBindBufferBase(0x90D2,index,b.bufferId())
            self.surface.bind()
            self.vertices.bind()
            self.surface.enableAttributeArray(0)
            self.surface.enableAttributeArray(1)
            self.surface.setAttributeBuffer(0,0x1406,0,3,32)
            self.surface.setAttributeBuffer(1,0x1406,16,3,32)
            self.vao.release()
        self.surface.release()
        self.asset_key = asset["key"]
        self.frame_key = None
        self.check("load")

    def deform(self, frame, key):
        if key == self.frame_key:
            return
        frame = np.ascontiguousarray(frame,dtype=np.float32)
        self.frame.bind()
        self.frame.write(0,frame.tobytes(),frame.nbytes)
        self.check("frame upload")
        if self.feedback:
            self.run_feedback(((self.skin,self.skin_feedback_inputs(),(self.positions,)),
                               (self.normals,self.normal_inputs,(self.normal_buffer,))))
            self.frame_key = key
            self.dispatches += 1
            return
        for index,b in enumerate((self.model,self.vertices,self.offsets,self.neighbors,self.frame)):
            self.gl.glBindBufferBase(0x90D2,index,b.bufferId())
        self.check("bind storage")
        for program in (self.skin,self.normals):
            program.bind()
            self.check("bind program")
            self.gl.glUniform1i(program.uniformLocation("vertexCount"),self.vertex_count)
            self.check("uniform")
            self.gl.glDispatchCompute((self.vertex_count+63)//64,1,1)
            self.check("compute")
            self.gl.glMemoryBarrier(0x2000 | 0x1)
            program.release()
        self.frame_key = key
        self.dispatches += 1

    def draw(self, eye, basis, width, height, color=(.71,.71,.71),transform=None,clear_depth=True):
        projection = QMatrix4x4()
        projection.perspective(40,width/max(1,height),.03,2000)
        view = np.eye(4)
        view[:3,:3] = basis * np.array([[1],[1],[-1]])
        view[:3,3] = -view[:3,:3] @ eye
        self.surface.bind()
        self.surface.setUniformValue("mvp",projection * QMatrix4x4(*view.reshape(-1).tolist()))
        self.surface.setUniformValue('sceneTransform',QMatrix4x4(*np.asarray(transform if transform is not None else np.eye(4)).reshape(-1).tolist()))
        self.surface.setUniformValue('bodyColor',QVector3D(*color))
        self.gl.glEnable(0x0B71)  # Depth test; true surface occlusion, not face sorting.
        self.gl.glDepthMask(True)
        self.gl.glDepthFunc(0x0201)
        self.gl.glDisable(0x0B90)
        self.gl.glDisable(0x0C11)
        self.gl.glDisable(0x0B44)
        if clear_depth:self.gl.glClear(0x100)
        self.vao.bind()
        self.gl.glDrawElements(0x4,self.index_count,0x1405,VoidPtr(0))
        self.vao.release()
        self.gl.glDisable(0x0B44)
        self.gl.glDisable(0x0B71)
        self.surface.release()

    def read_vertices(self):
        # Diagnostic only: a GPU readback would stall normal playback.
        if self.feedback:
            # Same (n, 8) position/normal rows as the interleaved compute buffer.
            parts = []
            for buffer in (self.positions,self.normal_buffer):
                part = np.empty((self.vertex_count,4),np.float32)
                buffer.bind()
                if not buffer.read(0,part,part.nbytes):
                    raise RuntimeError("Cannot read the verification buffer")
                parts.append(part)
            return np.concatenate(parts,axis=1)
        result = np.empty((self.vertex_count,8),np.float32)
        self.vertices.bind()
        if not self.vertices.read(0,result,result.nbytes):
            raise RuntimeError("Cannot read the verification buffer")
        return result

    def release_buffers(self):
        # Buffer textures must be freed while this context is current.
        for texture in self.views:
            texture.destroy()
        self.views.clear()
        for b in self.buffers:
            b.destroy()
        self.buffers.clear()
        self.asset_key = self.frame_key = None
        self.gpu_bytes = 0

    def close(self):
        self.release_buffers()
        self.vao.destroy()
        if self.feedback_vao is not None:
            self.feedback_vao.destroy()
        if not self.shared:
            for p in (self.skin,self.normals,self.surface):
                p.removeAllShaders()
