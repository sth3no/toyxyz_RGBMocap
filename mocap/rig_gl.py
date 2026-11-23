"""Bounded GPU rendering of target parts and untouched native mesh frames."""
import numpy as np
from shiboken6 import VoidPtr
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLBuffer,QOpenGLShader
from .mesh_gl import MeshRenderer,R32F,RGB32F,RGBA32F

SKIN='''#version 430 core
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Model { float model[]; };
struct Vertex { vec4 position; vec4 normal; };
layout(std430,binding=1) buffer Output { Vertex vertices[]; };
layout(std430,binding=4) readonly buffer Frame { float frame[]; };
uniform int vertexCount;
uniform int rawMode;
void main(){
 uint i=gl_GlobalInvocationID.x;if(i>=uint(vertexCount))return;
 if(rawMode!=0){vertices[i].position=vec4(frame[3*i],frame[3*i+1],frame[3*i+2],1);return;}
 int p=int(i)*7;int b=int(model[p+6])*16;
 vec4 v=vec4(model[p],model[p+1],model[p+2],1);
 vec4 n=vec4(model[p+3],model[p+4],model[p+5],0);
 vec4 a=vec4(frame[b],frame[b+1],frame[b+2],frame[b+3]);
 vec4 c=vec4(frame[b+4],frame[b+5],frame[b+6],frame[b+7]);
 vec4 d=vec4(frame[b+8],frame[b+9],frame[b+10],frame[b+11]);
 vertices[i].position=vec4(dot(a,v),dot(c,v),dot(d,v),1);
 vertices[i].normal=vec4(normalize(vec3(dot(a,n),dot(c,n),dot(d,n))),0);
}'''
# OpenGL 4.1 transform-feedback counterpart; see mesh_gl. Raw mode writes a
# placeholder normal that the following normals pass replaces.
FEEDBACK_SKIN='''#version 410 core
uniform samplerBuffer modelScalars;
uniform samplerBuffer poseRows;
uniform samplerBuffer rawPositions;
uniform int vertexCount;
uniform int rawMode;
out vec4 position;
out vec4 normal;
void main(){
 int i=gl_VertexID;
 if(rawMode!=0){position=vec4(texelFetch(rawPositions,i).xyz,1);normal=vec4(0);return;}
 int p=i*7;int b=int(texelFetch(modelScalars,p+6).r)*4;
 vec4 v=vec4(texelFetch(modelScalars,p).r,texelFetch(modelScalars,p+1).r,texelFetch(modelScalars,p+2).r,1);
 vec4 n=vec4(texelFetch(modelScalars,p+3).r,texelFetch(modelScalars,p+4).r,texelFetch(modelScalars,p+5).r,0);
 vec4 a=texelFetch(poseRows,b);
 vec4 c=texelFetch(poseRows,b+1);
 vec4 d=texelFetch(poseRows,b+2);
 position=vec4(dot(a,v),dot(c,v),dot(d,v),1);
 normal=vec4(normalize(vec3(dot(a,n),dot(c,n),dot(d,n))),0);
}'''


class RigRenderer(MeshRenderer):
    def __init__(self,shared=None):
        super().__init__(shared)
        self.topology_key=None
        if shared is None:
            self.skin.removeAllShaders()
            self.skin=self.kernel(SKIN,FEEDBACK_SKIN,('position','normal'))

    def load(self,asset):
        if self.asset_key==asset['key']:return
        topology=asset.get('topology_key',asset['key'])
        if self.buffers and self.topology_key==topology and self.vertex_count==int(asset['vertex_count']):
            # Same immutable topology, new motion generation. Keep shared GPU
            # data but force a new pose upload even at the same frame index.
            self.asset_key=asset['key'];self.frame_key=None;return
        self.release_buffers();self.vertex_count=int(asset['vertex_count']);self.index_count=asset['faces'].size
        self.raw=asset['mode']=='raw';self.bone_count=asset.get('bone_count',52);self.vao.bind()
        if self.shared and self.shared.topology_key!=topology:raise ValueError('Shared raw meshes must have identical topology')
        self.model=self.shared.model if self.shared else self.buffer(asset.get('packed',np.zeros(7,np.float32)))
        if self.feedback:self.feedback_outputs()
        else:self.vertices=self.buffer(self.vertex_count*32)
        self.offsets=self.shared.offsets if self.shared else self.buffer(asset.get('offsets',np.zeros(1,np.uint32)))
        self.neighbors=self.shared.neighbors if self.shared else self.buffer(asset.get('neighbors',np.zeros(1,np.uint32)))
        self.frame=self.buffer((self.vertex_count*3 if self.raw else self.bone_count*16)*4)
        self.indices=self.shared.indices if self.shared else self.buffer(asset['faces'],QOpenGLBuffer.IndexBuffer);self.indices.bind()
        if self.feedback:
            self.surface.bind();self.feedback_attributes(self.surface)
            self.skin_inputs=[('modelScalars',self.view(self.model,R32F)),('poseRows',self.view(self.frame,RGBA32F)),
                              ('rawPositions',self.view(self.frame,RGB32F))]
            self.normal_inputs=self.normal_views()
        else:
            self.surface.bind();self.vertices.bind()
            for i,offset in ((0,0),(1,16)):
                self.surface.enableAttributeArray(i);self.surface.setAttributeBuffer(i,0x1406,offset,3,32)
        self.vao.release();self.surface.release();self.asset_key=asset['key'];self.topology_key=topology;self.frame_key=None
        self.check('target load')

    def deform(self,frame,key):
        if self.frame_key==key:return
        frame=np.ascontiguousarray(frame,dtype=np.float32)
        expected=self.vertex_count*3 if self.raw else self.bone_count*16
        if frame.size!=expected or not np.isfinite(frame).all():raise ValueError('Invalid target render frame')
        self.frame.bind();self.frame.write(0,frame.tobytes(),frame.nbytes)
        if self.feedback:
            passes=[(self.skin,self.skin_inputs,(self.positions,self.normal_buffer))]
            if self.raw:passes.append((self.normals,self.normal_inputs,(self.normal_buffer,)))
            self.run_feedback(passes,uniforms=(('rawMode',int(self.raw)),))
            self.check('target deform');self.frame_key=key;self.dispatches+=1;return
        for index,buffer in enumerate((self.model,self.vertices,self.offsets,self.neighbors,self.frame)):
            self.gl.glBindBufferBase(0x90D2,index,buffer.bufferId())
        for program in ([self.skin,self.normals] if self.raw else [self.skin]):
            program.bind();self.gl.glUniform1i(program.uniformLocation('vertexCount'),self.vertex_count)
            if program is self.skin:self.gl.glUniform1i(program.uniformLocation('rawMode'),int(self.raw))
            self.gl.glDispatchCompute((self.vertex_count+63)//64,1,1);self.gl.glMemoryBarrier(0x2000|0x1);program.release()
        self.check('target deform');self.frame_key=key;self.dispatches+=1
