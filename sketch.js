let resx, resy;

// vertex shader 
src_v = `#version 300 es
precision highp float;

in vec2 a;
out vec2 u;

void main() {
  u = a * 2. - 1.;
  gl_Position = vec4(u,0,1);
}`;

// fragment shader for main canvas
src_f = `#version 300 es
precision highp float;

in vec2 u;
out vec4 cc;
uniform vec2 res;
uniform float time;
uniform sampler2D jfa;

void main() {
    vec2 uv = u * .5 + .5;    
    cc = texture(jfa,uv) * vec4(u.x,u.y,.5, 1.);
}`;

//fragment shader for framebuffer
src_buffer = `#version 300 es
precision highp float;

in vec2 u;
out vec4 cc;
uniform vec2 res;
uniform sampler2D glyph;

void main() {
    vec2 uv = abs(u);

    // cc = texture(glyph,uv) * vec4(u.x,u.y,.5, 1.);
    cc = vec4(u.x,u.y,.5,1.);
}`;

// make a canvas and gl context
C = ({body} = D = document).createElement('canvas');
body.appendChild(C);
gl = C.getContext('webgl2');

//set dimensions
let w = innerWidth,
  h = innerHeight,
  dpr = devicePixelRatio;
let minRes = Math.min(w, h);
h = w = minRes * .75;
resx = C.width = w * dpr | 0;
resy = C.height = h * dpr | 0;
C.style.width = w + 'px';
C.style.height = h + 'px';

//function to create shader
Shader = (typ, src) => {
  const s = gl.createShader(typ);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  return s;
}

//function to create program
createProgram = (vertex, fragment) => {
  let program = gl.createProgram();
  vs = Shader(gl.VERTEX_SHADER, vertex);
  fs = Shader(gl.FRAGMENT_SHADER, fragment);
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.log(`Link failed:\n${gl.getProgramInfoLog(program)}`);
    console.log(`VS LOG:\n${gl.getShaderInfoLog(vs)}`);
    console.log(`FS LOG:\n${gl.getShaderInfoLog(fs)}`);
    throw 'AARG DED';
  }
  return program;
}


//////load glyph into texture
//create texture variable for glyph

// var glyph_tex = gl.createTexture();
// gl.activeTexture(gl.TEXTURE0 + 1);
// gl.bindTexture(gl.TEXTURE_2D, glyph_tex);
// gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
// gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
// gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
// gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
// //temporarily make texture black
// gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));


// //load real texture
// var glyphImage = new Image();
// glyphImage.src = "glyph.png";
// glyphImage.addEventListener('load', function () {
//   gl.bindTexture(gl.TEXTURE_2D, glyph_tex);
//   gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, glyphImage);
//   gl.generateMipmap(gl.TEXTURE_2D);
// });


//create the buffer program
bufferProgram = createProgram(src_v, src_buffer);

////////////////////load glyph into the frame buffer
//create buffer texture
var bufferTex = gl.createTexture();
gl.activeTexture(gl.TEXTURE0 + 0);
gl.bindTexture(gl.TEXTURE_2D, bufferTex);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, resx, resy, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

//create framebuffer
const fb = gl.createFramebuffer();
gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
const attachmentPoint = gl.COLOR_ATTACHMENT0;
gl.framebufferTexture2D(gl.FRAMEBUFFER, attachmentPoint, gl.TEXTURE_2D, bufferTex, 0);
gl.viewport(0, 0, resx, resy);


gl.useProgram(bufferProgram);
loc_buffer_res = gl.getUniformLocation(bufferProgram, 'res');
loc_glyph = gl.getUniformLocation(bufferProgram, 'glyph');
gl.uniform1i(loc_glyph, 0);
gl.uniform2f(loc_buffer_res, resx, resy);

gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
gl.bufferData(gl.ARRAY_BUFFER, Float32Array.of(0, 1, 0, 0, 1, 1, 1, 0), gl.STATIC_DRAW);
let loc_a = gl.getAttribLocation(bufferProgram, 'a');
gl.enableVertexAttribArray(loc_a);
gl.vertexAttribPointer(loc_a, 2, gl.FLOAT, false, 0, 0);
gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);




////////////////////main program
gl.bindFramebuffer(gl.FRAMEBUFFER, null);
mainProgram = createProgram(src_v, src_f);
gl.viewport(0, 0, resx, resy);
// get the uniform locs
loc_res = gl.getUniformLocation(mainProgram, 'res');
loc_time = gl.getUniformLocation(mainProgram, 'time');

// This loads a bunch of coordinates and connects them to the`a`-attribute.


loc_jfa = gl.getUniformLocation(mainProgram, 'jfa');

gl.useProgram(mainProgram);
gl.uniform1i(loc_jfa, 1);
let time = performance.now() * .001;
gl.uniform1f(loc_time, time);
gl.uniform2f(loc_res, resx, resy);
gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
gl.bufferData(gl.ARRAY_BUFFER, Float32Array.of(0, 1, 0, 0, 1, 1, 1, 0), gl.STATIC_DRAW);
loc_a = gl.getAttribLocation(mainProgram, 'a');
gl.enableVertexAttribArray(loc_a);
gl.vertexAttribPointer(loc_a, 2, gl.FLOAT, false, 0, 0);
gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);






function render() {




  requestAnimationFrame(render);
}


render();