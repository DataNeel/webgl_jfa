let firstframe = true;
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
    vec2 uv = (u * .5 + .5);
    vec4 t = texture(jfa,uv);
    float dist = sin(distance(vec2(t.x,t.y),uv)*100.-time*5.);
    // cc = vec4(dist, dist, dist, 1.0);
    cc = t;
    // cc = vec4(uv.x,uv.y,0.,1.);
}`;

//fragment shader for init
src_init = `#version 300 es
precision highp float;

in vec2 u;
out ivec2 cc;
uniform vec2 res;
uniform sampler2D glyph;

void main() {
    // ivec2 tc = ivec2(gl_FragCoord+.5);
    
    vec4 t =  texelFetch(glyph,ivec2(gl_FragCoord.xy)/2,1);
    cc = int(t.x) * ivec2(gl_FragCoord.x,gl_FragCoord.y);
    
}`;

//fragment shader for ping
src_ping = `#version 300 es
precision highp float;

in vec2 u;

out ivec2 cc;
uniform vec2 res;
uniform sampler2D pong;
uniform float jfa_step;

void main() {
    ivec2 tc = ivec2(gl_FragCoord.xy);
    int offset = int(jfa_step+.5);
    ivec2 t = ivec2(-1);
    float bestDist = 99999.;
    for (int i = -1; i <=1; i++) {
        for (int j = -1; j <=1; j++) {
            ivec2 tc2 = tc+ivec2(i*offset,j*offset);
            ivec2 t2 = ivec2(texelFetch(pong,tc2,0).xy);
            float new_dist = distance(vec2(tc2),vec2(tc));
            if (t2.x>0 && t2.y>0 && new_dist<bestDist) {
                t = t2;
                bestDist = new_dist;
            }
        }
    }
    cc = t;
}`;


let steps_override = 3.;


//fragment shader for pong
src_pong = `#version 300 es
precision highp float;

in vec2 u;
out ivec2 cc;
uniform vec2 res;
uniform sampler2D ping;

void main() {
    ivec2 tc = ivec2(gl_FragCoord.xy);
    cc = ivec2(texelFetch(ping,tc,0).xy);
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
    gl.getExtension('OES_texture_float_linear');
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);

    //set dimensions
    let resx, resy;
    let w = innerWidth,
    h = innerHeight,
    dpr = devicePixelRatio;
    let minRes = Math.min(w, h);
    h = w = minRes * 1.;
    resx = C.width = w * dpr | 0;
    resy = C.height = h * dpr | 0;
    // resx, resy = 1024;
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
    // glyphImage.src = "glyph.png";
    glyphImage.src = "glyph.png";
    glyphImage.addEventListener('load', function () {
    gl.activeTexture(gl.TEXTURE0+0);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, glyphImage);
    gl.generateMipmap(gl.TEXTURE_2D);
    });

//init program
    //set up the init program
    initProgram = createProgram(src_v, src_init);
    loc_res_init = gl.getUniformLocation(initProgram, 'res');
    loc_glyph_init = gl.getUniformLocation(initProgram, 'glyph');
    loc_a_init = gl.getAttribLocation(initProgram, 'a');
    gl.useProgram(initProgram);
    gl.uniform1i(loc_glyph_init,0);

//ping program
    //set up the ping program
    pingProgram = createProgram(src_v, src_ping);
    loc_res_ping = gl.getUniformLocation(pingProgram, 'res');
    loc_pong_ping = gl.getUniformLocation(pingProgram, 'pong');
    loc_a_ping = gl.getAttribLocation(pingProgram, 'a');
    loc_step_ping = gl.getUniformLocation(pingProgram,'jfa_step');
    gl.useProgram(pingProgram);
    gl.uniform1i(loc_pong_ping,0);

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
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG16I, resx, resy, 0, gl.RG_INTEGER, gl.SHORT, null);
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
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG16I, resx, resy, 0, gl.RG_INTEGER, gl.SHORT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, pongAttachmentPoint, gl.TEXTURE_2D, pongTex, 0);


 

   var then = 0;
    requestAnimationFrame(drawScene);

    function drawScene(time) {
        // convert to seconds
        time *= 0.001;
        // Subtract the previous time from the current time
        var deltaTime = time - then;
        // Remember the current time for the next frame.
        then = time;

        //ping
        gl.bindFramebuffer(gl.FRAMEBUFFER, ping);
        
        gl.viewport(0,0,resx,resy);
        gl.clearColor(0, 0, 1, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.useProgram(initProgram);
        gl.uniform1i(loc_glyph_init,0);
        
        gl.bindVertexArray(vao_ping);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        firstframe && console.log(resx);
        let steps = Math.ceil(Math.log2(resx));
        firstframe && console.log("steps: "+steps);
        steps = steps_override;
        for (let i = 0; i <steps; i++) {
            //pong
            gl.bindFramebuffer(gl.FRAMEBUFFER,pong);
            gl.viewport(0,0,resx,resy);
            gl.clearColor(0, 0, 1, 1);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
            gl.useProgram(pongProgram);
            gl.uniform1i(loc_ping_pong,1);
            gl.bindVertexArray(vao_pong);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            //ping
            gl.bindFramebuffer(gl.FRAMEBUFFER,ping);
            gl.viewport(0,0,resx,resy);
            gl.clearColor(0, 0, 1, 1);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
            gl.useProgram(pingProgram);
            gl.uniform1i(loc_pong_ping,2);
            let step = 2**(Math.log2(resx)-i-1);
            firstframe && console.log(step);
            gl.uniform1f(loc_step_ping,step);
            gl.bindVertexArray(vao_ping);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        }

    
        //main canvas stuffs
        // gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        // gl.viewport(0,0,resx,resy);
        // gl.clearColor(0, 0, 1, 1);
        // gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        // gl.useProgram(mainProgram);
        // gl.uniform1f(loc_time_main,time);
        // gl.bindVertexArray(vao_main);
        // gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        firstframe = false;
        requestAnimationFrame(drawScene);
    }

    
}

main();