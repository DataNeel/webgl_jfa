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
    vec2 uv = abs(u);
    cc = texture(jfa,uv);
}`;

//fragment shader for framebuffer
src_buffer = `#version 300 es
precision highp float;

in vec2 u;
out vec4 cc;
uniform vec2 res;
uniform sampler2D glyph;

void main() {
    vec2 uv = u * .5 + .5;    
    

    cc = texture(glyph,uv) * vec4(u.x,u.y,.5, 1.);
    // cc = vec4(u.x,u.y,.5,1.);
}`;

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

function main() {
    //make canvas
    C = ({body} = D = document).createElement('canvas');
    body.appendChild(C);
    gl = C.getContext('webgl2');
    gl.getExtension('EXT_color_buffer_float');
    gl.getExtension('OES_texture_float_linear')

    //set dimensions
    let resx, resy;
    let w = innerWidth,
    h = innerHeight,
    dpr = devicePixelRatio;
    let minRes = Math.min(w, h);
    h = w = minRes * .75;
    resx = C.width = w * dpr | 0;
    resy = C.height = h * dpr | 0;
    C.style.width = w + 'px';
    C.style.height = h + 'px';

    let positions = Float32Array.of(0, 1, 0, 0, 1, 1, 1, 0);

    ////load glyph into texture
    // create texture variable for glyph

    var glyph_tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + 1);
    gl.bindTexture(gl.TEXTURE_2D, glyph_tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    //temporarily make texture black
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));


    //load real texture
    var glyphImage = new Image();
    glyphImage.src = "glyph.png";
    glyphImage.addEventListener('load', function () {
    gl.bindTexture(gl.TEXTURE_2D, glyph_tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, glyphImage);
    gl.generateMipmap(gl.TEXTURE_2D);
    });

    //set up the buffer program
    framebufferProgram = createProgram(src_v, src_buffer);
    loc_res_framebuffer = gl.getUniformLocation(framebufferProgram, 'res');
    loc_glyph_framebuffer = gl.getUniformLocation(framebufferProgram, 'glyph');
    loc_a_framebuffer = gl.getAttribLocation(framebufferProgram, 'a');

    //create a buffer and vao for framebuffer
    positionBuffer_framebuffer = gl.createBuffer();
    vao_framebuffer = gl.createVertexArray();
    gl.bindVertexArray(vao_framebuffer);
    gl.enableVertexAttribArray(loc_a_framebuffer);
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer_framebuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.vertexAttribPointer(loc_a_framebuffer, 2, gl.FLOAT, false, 0, 0);

    //set up main canvas program
    mainProgram = createProgram(src_v, src_f);
    loc_res_main = gl.getUniformLocation(mainProgram, 'res');
    loc_time_main = gl.getUniformLocation(mainProgram, 'time');
    loc_buffer_main = gl.getUniformLocation(mainProgram, 'jfa');
    loc_a_main = gl.getAttribLocation(mainProgram, 'a');

    //create a buffer and vao for main canvas
    positionBuffer_main = gl.createBuffer();
    vao_main = gl.createVertexArray();
    gl.bindVertexArray(vao_main);
    gl.enableVertexAttribArray(loc_a_main);
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer_main);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.vertexAttribPointer(loc_a_main, 2, gl.FLOAT, false, 0, 0);


    ////////////////////load glyph into the frame buffer
    //create buffer texture
    var framebufferTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + 0);
    gl.bindTexture(gl.TEXTURE_2D, framebufferTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, resx, resy, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    //create framebuffer
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    const attachmentPoint = gl.COLOR_ATTACHMENT0;
    gl.framebufferTexture2D(gl.FRAMEBUFFER, attachmentPoint, gl.TEXTURE_2D, framebufferTex, 0);
    
    var then = 0;
    requestAnimationFrame(drawScene);

    function drawScene(time) {
        // convert to seconds
        time *= 0.001;
        // Subtract the previous time from the current time
        var deltaTime = time - then;
        // Remember the current time for the next frame.
        then = time;

        //framebuffer stuffs
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
        gl.bindTexture(gl.TEXTURE_2D,glyph_tex);
        gl.viewport(0,0,resx,resy);
        gl.clearColor(0, 0, 1, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.useProgram(framebufferProgram);
        gl.bindVertexArray(vao_framebuffer);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    
        //main canvas stuffs
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.bindTexture(gl.TEXTURE_2D, framebufferTex);
        gl.viewport(0,0,resx,resy);
        gl.clearColor(0, 0, 1, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.useProgram(mainProgram);
        gl.bindVertexArray(vao_main);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        requestAnimationFrame(drawScene);
    }

    
}

main();