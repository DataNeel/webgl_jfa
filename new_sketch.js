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

//fragment shader for ping
src_ping = `#version 300 es
precision highp float;

in vec2 u;
out vec4 cc;
uniform vec2 res;
uniform sampler2D glyph;

void main() {
    vec2 uv = u * .5 + .5;
    vec2 onePixel = vec2(50) / vec2(textureSize(glyph, 0));
    
    vec4 t = texture(glyph,uv) + texture(glyph,uv+vec2(onePixel));
    cc = t * vec4(u.x,u.y,.5, 1.);
    // cc = vec4(u.x,u.y,.5,1.);
}`;

//fragment shader for pong
src_pong = `#version 300 es
precision highp float;

in vec2 u;
out vec4 cc;
uniform vec2 res;
uniform sampler2D ping;

void main() {
    vec2 uv = u * .5 + .5;
    cc = texture(ping,uv);
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
    gl.activeTexture(gl.TEXTURE0 + 0);
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
    gl.activeTexture(gl.TEXTURE0+0);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, glyphImage);
    gl.generateMipmap(gl.TEXTURE_2D);
    });

//ping program
    //set up the ping program
    pingProgram = createProgram(src_v, src_ping);
    loc_res_ping = gl.getUniformLocation(pingProgram, 'res');
    loc_glyph_ping = gl.getUniformLocation(pingProgram, 'glyph');
    loc_a_ping = gl.getAttribLocation(pingProgram, 'a');
    gl.useProgram(pingProgram);
    gl.uniform1i(loc_glyph_ping,0);

    //create a buffer and vao for ping
    positionBuffer_ping = gl.createBuffer();
    vao_ping = gl.createVertexArray();
    gl.bindVertexArray(vao_ping);
    gl.enableVertexAttribArray(loc_a_ping);
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer_ping);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.vertexAttribPointer(loc_a_ping, 2, gl.FLOAT, false, 0, 0);

//pong program
    //set up the pong program
    pongProgram = createProgram(src_v, src_pong);
    loc_res_pong = gl.getUniformLocation(pongProgram, 'res');
    loc_ping_pong = gl.getUniformLocation(pongProgram, 'ping');
    loc_a_pong = gl.getAttribLocation(pongProgram, 'a');
    gl.useProgram(pongProgram);
    gl.uniform1i(loc_ping_pong,1);

    //create a buffer and vao for ping
    positionBuffer_pong = gl.createBuffer();
    vao_pong = gl.createVertexArray();
    gl.bindVertexArray(vao_pong);
    gl.enableVertexAttribArray(loc_a_pong);
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer_pong);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.vertexAttribPointer(loc_a_pong, 2, gl.FLOAT, false, 0, 0);


//main program
    //set up main canvas program
    mainProgram = createProgram(src_v, src_f);
    gl.useProgram(mainProgram);
    loc_res_main = gl.getUniformLocation(mainProgram, 'res');
    loc_time_main = gl.getUniformLocation(mainProgram, 'time');
    loc_buffer_main = gl.getUniformLocation(mainProgram, 'jfa');
    loc_a_main = gl.getAttribLocation(mainProgram, 'a');
    gl.uniform1i(loc_buffer_main,1);
    

    //create a buffer and vao for main canvas
    positionBuffer_main = gl.createBuffer();
    vao_main = gl.createVertexArray();
    gl.bindVertexArray(vao_main);
    gl.enableVertexAttribArray(loc_a_main);
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer_main);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.vertexAttribPointer(loc_a_main, 2, gl.FLOAT, false, 0, 0);




//ping fbo and tex
    //create ping framebuffer
    const ping = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, ping);
    const pingAttachmentPoint = gl.COLOR_ATTACHMENT0;

    //create ping texture
    var pingTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + 1);
    gl.bindTexture(gl.TEXTURE_2D, pingTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, resx, resy, 0, gl.RGBA, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, pingAttachmentPoint, gl.TEXTURE_2D, pingTex, 0);


//pong fbo and tex
    //create pong framebuffer
    const pong = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, pong);
    const pongAttachmentPoint = gl.COLOR_ATTACHMENT0;
    
    

    //create pong texture
    var pongTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + 2);
    gl.bindTexture(gl.TEXTURE_2D, pongTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, resx, resy, 0, gl.RGBA, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, pingAttachmentPoint, gl.TEXTURE_2D, pongTex, 0);


 

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
        gl.bindFramebuffer(gl.FRAMEBUFFER, ping);
        
        gl.viewport(0,0,resx,resy);
        gl.clearColor(0, 0, 1, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.useProgram(pingProgram);
        
        gl.bindVertexArray(vao_ping);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        for (let i = 0; i < 10; i++) {
            
        }

    
        //main canvas stuffs
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
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