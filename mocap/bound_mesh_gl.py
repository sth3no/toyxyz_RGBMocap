"""Small canonical actor binding over the shared registered SMPL-X buffers."""
import numpy as np
from PySide6.QtOpenGL import QOpenGLShader
from .mesh_gl import MeshRenderer,RGB32F

SKIN="""#version 430 core
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Model { float model[]; };
struct Vertex { vec4 position; vec4 normal; };
layout(std430,binding=1) buffer Output { Vertex vertices[]; };
layout(std430,binding=4) readonly buffer Frame { float pose[]; };
layout(std430,binding=5) readonly buffer Binding { float bindData[]; };
uniform int vertexCount;
void main() {
 uint i=gl_GlobalInvocationID.x;
 if(i>=uint(vertexCount)) return;
 int n=vertexCount*3;
 vec3 delta=vec3(0);
 for(int k=0;k<486;k++) {
  int o=n*21+k*n+int(3*i);
  delta+=pose[880+k]*vec3(model[o],model[o+1],model[o+2]);
 }
 vec3 transported=vec3(0);
 for(int j=0;j<55;j++) {
  float w=model[n*507+int(i)*55+j];
  if(w==0) continue;
  int o=n+j*9;
  transported+=w*vec3(dot(vec3(bindData[o],bindData[o+1],bindData[o+2]),delta),
   dot(vec3(bindData[o+3],bindData[o+4],bindData[o+5]),delta),
   dot(vec3(bindData[o+6],bindData[o+7],bindData[o+8]),delta));
 }
 vec4 p=vec4(vec3(bindData[3*i],bindData[3*i+1],bindData[3*i+2])+transported,1);
 vec3 result=vec3(0);
 for(int j=0;j<55;j++) {
  float w=model[n*507+int(i)*55+j];
  if(w==0) continue;
  int o=j*16;
  result+=w*vec3(dot(vec4(pose[o],pose[o+1],pose[o+2],pose[o+3]),p),
   dot(vec4(pose[o+4],pose[o+5],pose[o+6],pose[o+7]),p),
   dot(vec4(pose[o+8],pose[o+9],pose[o+10],pose[o+11]),p));
 }
 vertices[i].position=vec4(result,1);
}
"""
# OpenGL 4.1 transform-feedback counterpart; see mesh_gl. bindVectors is an
# RGB32F view of the binding: vertex i at texel i, frame j row r at c+3j+r.
FEEDBACK_SKIN="""#version 410 core
uniform samplerBuffer modelVectors;
uniform samplerBuffer modelScalars;
uniform samplerBuffer poseRows;
uniform samplerBuffer poseScalars;
uniform samplerBuffer bindVectors;
uniform int vertexCount;
out vec4 position;
void main() {
 int i=gl_VertexID;
 int c=vertexCount;
 vec3 delta=vec3(0);
 for(int k=0;k<486;k++)
  delta+=texelFetch(poseScalars,880+k).r*texelFetch(modelVectors,(21+k)*c+i).xyz;
 vec3 transported=vec3(0);
 for(int j=0;j<55;j++) {
  float w=texelFetch(modelScalars,c*3*507+i*55+j).r;
  if(w==0) continue;
  int o=c+3*j;
  transported+=w*vec3(dot(texelFetch(bindVectors,o).xyz,delta),
   dot(texelFetch(bindVectors,o+1).xyz,delta),
   dot(texelFetch(bindVectors,o+2).xyz,delta));
 }
 vec4 p=vec4(texelFetch(bindVectors,i).xyz+transported,1);
 vec3 result=vec3(0);
 for(int j=0;j<55;j++) {
  float w=texelFetch(modelScalars,c*3*507+i*55+j).r;
  if(w==0) continue;
  result+=w*vec3(dot(texelFetch(poseRows,4*j),p),
   dot(texelFetch(poseRows,4*j+1),p),
   dot(texelFetch(poseRows,4*j+2),p));
 }
 position=vec4(result,1);
}
"""


class BoundMeshRenderer(MeshRenderer):
    def __init__(self,shared=None):
        self.binding_buffer=None;self.binding_key=None;self.binding_view=None
        super().__init__(shared)
        if shared is None:
            self.skin.removeAllShaders()
            self.skin=self.kernel(SKIN,FEEDBACK_SKIN)

    def release_buffers(self):
        super().release_buffers()
        self.binding_buffer=None;self.binding_key=None;self.binding_view=None

    def skin_feedback_inputs(self):
        return self.skin_inputs+[('bindVectors',self.binding_view)]

    def bind_skin(self,binding):
        key=binding.get('key',id(binding))
        if key==self.binding_key:return
        value=np.concatenate((binding['vertices'].reshape(-1),binding['frames'].reshape(-1))).astype(np.float32)
        if self.binding_buffer is None:
            self.binding_buffer=self.buffer(value)
            if self.feedback:self.binding_view=self.view(self.binding_buffer,RGB32F)
        else:
            self.binding_buffer.bind()
            if self.binding_buffer.size()!=value.nbytes:raise ValueError('Canonical binding dimensions changed')
            self.binding_buffer.write(0,value.tobytes(),value.nbytes)
        self.binding_key=key;self.frame_key=None

    def deform(self,frame,key):
        if self.binding_buffer is None:raise ValueError('Canonical skin binding missing')
        if not self.feedback:self.gl.glBindBufferBase(0x90D2,5,self.binding_buffer.bufferId())
        super().deform(frame,(self.binding_key,key))
