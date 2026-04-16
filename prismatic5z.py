# Enhanced prismatic5y.py
import sys, time
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import compileProgram, compileShader

# ---------- Globals ----------
window = None
shader = None
vao = None

NUM_SUPER = 32768
NUM_INSTANCES = 4096
MAX_SLICE = 8

VIRT_WIDTH = 7680   # Virtual canvas width (8K)
VIRT_HEIGHT = 4320
FPS = 60.0

fibTable = np.array([1,1,2,3,5,8,13,21,34,55,89,144]*12, dtype=np.float32)[:128]
primeTable = np.array([2,3,5,7,11,13,17,19,23,29,31,37]*12, dtype=np.float32)[:128]

# ---------- Shader ----------
FRAGMENT_SRC = """
#version 330
in vec2 texCoord;
out vec4 fragColor;

uniform float cycle;
uniform float omegaTime;
uniform float phi;
uniform float phiInv;
uniform int instanceID;
uniform int NUM_SUPER;
uniform int NUM_INSTANCES;
uniform int MAX_SLICE;

uniform float fibTable[128];
uniform float primeTable[128];

float prismatic_recursion(int id, float r){
    float phi_harm = pow(phi, float(mod(id,16)));
    float fib_harm = fibTable[id % 128];
    float dyadic = float(1 << int(mod(float(id),16.0)));
    float prime_harm = primeTable[id % 128];
    float Omega = 0.5 + 0.5*sin(omegaTime + float(id)*0.01);
    float r_dim = pow(r, float(mod(id,7)+1));
    return sqrt(phi_harm * fib_harm * dyadic * prime_harm * Omega) * r_dim;
}

void main(){
    float r = length(texCoord - 0.5) * 2.0;
    float val = 0.0;

    for(int s=0; s<NUM_SUPER; s++){
        int idx = (instanceID * NUM_SUPER + s) % NUM_INSTANCES;
        val += prismatic_recursion(idx, r);
    }
    val /= float(NUM_SUPER);

    float phase = sin(cycle*0.01 + val);
    float slice = mod(float(instanceID), float(MAX_SLICE));
    fragColor = vec4(val, phase, r, slice/float(MAX_SLICE));
}
"""

VERTEX_SRC = """
#version 330
in vec2 position;
out vec2 texCoord;
void main(){
    texCoord = 0.5*(position+1.0);
    gl_Position = vec4(position,0.0,1.0);
}
"""

# ---------- OpenGL Setup ----------
def init_gl():
    global shader, vao
    shader = compileProgram(
        compileShader(VERTEX_SRC, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SRC, GL_FRAGMENT_SHADER)
    )
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vertices = np.array([-1,-1, 1,-1, -1,1, 1,1], dtype=np.float32)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    pos = glGetAttribLocation(shader, "position")
    glEnableVertexAttribArray(pos)
    glVertexAttribPointer(pos, 2, GL_FLOAT, GL_FALSE, 0, None)

# ---------- Render Loop ----------
cycle = 0.0
tileX = 0
tileY = 0
tilesX = VIRT_WIDTH // 1920
tilesY = VIRT_HEIGHT // 1080

def display():
    global cycle, tileX, tileY
    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(shader)

    # Dynamic instance mapping to virtual canvas tiles
    instance_id = tileY * tilesX + tileX

    glUniform1f(glGetUniformLocation(shader,"cycle"), cycle)
    glUniform1f(glGetUniformLocation(shader,"omegaTime"), time.time())
    glUniform1f(glGetUniformLocation(shader,"phi"), 1.61803398875)
    glUniform1f(glGetUniformLocation(shader,"phiInv"), 0.61803398875)
    glUniform1i(glGetUniformLocation(shader,"NUM_SUPER"), NUM_SUPER)
    glUniform1i(glGetUniformLocation(shader,"NUM_INSTANCES"), NUM_INSTANCES)
    glUniform1i(glGetUniformLocation(shader,"MAX_SLICE"), MAX_SLICE)
    glUniform1i(glGetUniformLocation(shader,"instanceID"), instance_id)
    glUniform1fv(glGetUniformLocation(shader,"fibTable"), 128, fibTable)
    glUniform1fv(glGetUniformLocation(shader,"primeTable"), 128, primeTable)

    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
    glutSwapBuffers()

    # Advance waterfall tiles
    tileX += 1
    if tileX >= tilesX:
        tileX = 0
        tileY += 1
        if tileY >= tilesY:
            tileY = 0
    cycle += 1.0

def idle():
    glutPostRedisplay()

# ---------- Main ----------
def main():
    global window
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(1920,1080)  # Display window
    glutInitWindowPosition(50,50)
    glutCreateWindow(b"HDGL Prismatic BaseINFINITE Fabric - 60+Hz + 8K Waterfall")
    init_gl()
    glutDisplayFunc(display)
    glutIdleFunc(idle)
    glutMainLoop()

if __name__ == "__main__":
    main()
