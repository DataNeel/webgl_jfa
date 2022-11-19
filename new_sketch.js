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


vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}

float snoise(vec3 v){ 
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //  x0 = x0 - 0. + 0.0 * C 
  vec3 x1 = x0 - i1 + 1.0 * C.xxx;
  vec3 x2 = x0 - i2 + 2.0 * C.xxx;
  vec3 x3 = x0 - 1. + 3.0 * C.xxx;

// Permutations
  i = mod(i, 289.0 ); 
  vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients
// ( N*N points uniformly over a square, mapped onto an octahedron.)
  float n_ = 1.0/7.0; // N=7
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                dot(p2,x2), dot(p3,x3) ) );
}

void main() {
    vec2 uv = gl_FragCoord.xy/res;
    vec4 t = texture(jfa,uv);
    // uv.x *= uv.y/uv.x;
    float d1 = distance(vec2(t.x,t.y),uv);
    float d2 = distance(vec2(t.b,t.a),uv);
    // d2 = d1;
    float d = .005;
    float scale = .25;
    float n1 = d + snoise(vec3(uv*50.,time*.5))*d*scale;
    float n2 = d + snoise(vec3(uv*30.,-time*.5))*d*scale;
    // n1 = .005;
    

    float space = 15.*max(d1*d2,.002);
    float width = space*.15;
    // d1 *= .7;
    // d2 *= .7;
    float dist = step(mod(d1,space),n1+width)-step(mod(d1,space),n1-width);
    float dist2 = step(mod(d2,space),n2+width)-step(mod(d2,space),n2-width*1.5);
    
    // float dist = smoothstep(n1,n1+width*2.,mod(d1,space))-smoothstep(n1,n1-width*2.,mod(d1,space));
    // float dist2 = smoothstep(n2,n2+width*2.,mod(d2,space))-smoothstep(n2,n2-width*2.,mod(d2,space));
    



    vec4 c1 = vec4(6, 123, 194,255.)/255.;
    vec4 c2 = vec4(213, 96, 98,255.)/255.;
    vec4 c3 = vec4(243, 119, 72,255.)/255.;
    vec4 c4 = vec4(236, 195, 11,255.)/255.;
    
    vec4 ca = mix(c1,c2,dist);
    vec4 cb = mix(c3,c4,dist);
    cc = mix(ca,cb,dist2);


    // cc = vec4(t.b,t.a,1.,1.);
}`;
//fragment shader for init
src_init = `#version 300 es
precision highp float;

in vec2 u;
out vec4 cc;
uniform vec2 res;
uniform sampler2D glyph;

void main() {
    vec2 uv = u * .5 + .5;
    cc=vec4(999.);
    vec4 t = texture(glyph,uv);
    float r = step(.1,t.r);
    float g = step(.1,t.g);
    if(r>0.1)cc=vec4(uv,cc.ba);
    if(g>0.1)cc=vec4(cc.rg,uv);
}`;

//fragment shader for ping
src_ping = `#version 300 es
precision highp float;
in vec2 u;
out vec4 cc;
uniform vec2 res;
uniform sampler2D pong;
uniform float jfa_step;
uniform float frame;

void main() {
    vec2 uv = gl_FragCoord.xy/res;
    
   vec2 onePixel = vec2(.5) / pow(2.,frame);
   vec4 t = vec4(0.);
   float bestDistR = 99999.;
   float bestDistG = 99999.;
   for (float i = -1.; i <=1.; i++) {
       for (float j = -1.; j <=1.; j++) {
           vec2 uv_s = fract(uv+vec2(i,j)*onePixel);
           vec4 t2 = texelFetch(pong,ivec2(uv_s*res),0);
           float new_dist = length(t2.xy-uv);
           if (new_dist<bestDistR) {
               t.rg = t2.rg;
               bestDistR = new_dist;
           }
           new_dist = length(t2.ba-uv);
           if (new_dist<bestDistG) {
                t.ba = t2.ba;
                bestDistG = new_dist;
           }
       }
   }
	cc=t;//+texture(pong, fract(uv));
	// cc/=2.;
   // cc = ;
////cc.a=1.;
}`;
let steps_override =20.;

//fragment shader for pong
src_pong = `#version 300 es
precision highp float;

in vec2 u;
out vec4 cc;
uniform vec2 res;
uniform sampler2D ping;

void main() {
    vec2 uv = gl_FragCoord.xy/res;
    cc = texture(ping,uv);
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
    h = w =  minRes * 1.;
    // w = h * 16/9;
    // h = w / 16 * 9;
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
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 100, 100, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([...Array(100**2)].map(_=>[0, 0, 0, 255]).flat()));

    //load real texture
    var glyphImage = new Image();
    glyphImage.src = "rocks6.png";
    glyphImage.addEventListener('load', function () {
    gl.activeTexture(gl.TEXTURE0+0);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, glyphImage);
    gl.generateMipmap(gl.TEXTURE_2D);



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
         steps = Math.min(steps,steps_override);
			let frame=0
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
            gl.uniform1f(loc_frame_ping,frame);
            gl.bindVertexArray(vao_ping);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

					frame++
        }


    });

//init program
    //set up the init program
    initProgram = createProgram(src_v, src_init);
    loc_res_init = gl.getUniformLocation(initProgram, 'res');
		gl.useProgram(initProgram)
    gl.uniform2f(loc_res_init,resx,resy);
    loc_glyph_init = gl.getUniformLocation(initProgram, 'glyph');
    loc_a_init = gl.getAttribLocation(initProgram, 'a');
    gl.useProgram(initProgram);
    gl.uniform1i(loc_glyph_init,0);

//ping program
    //set up the ping program
    pingProgram = createProgram(src_v, src_ping);
    loc_res_ping = gl.getUniformLocation(pingProgram, 'res');
		gl.useProgram(pingProgram)
		gl.uniform2f(loc_res_ping,resx,resy);
    loc_pong_ping = gl.getUniformLocation(pingProgram, 'pong');
    loc_a_ping = gl.getAttribLocation(pingProgram, 'a');
    loc_step_ping = gl.getUniformLocation(pingProgram,'jfa_step');
    loc_frame_ping = gl.getUniformLocation(pingProgram,'frame');
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
		gl.useProgram(pongProgram)
    gl.uniform2f(loc_res_pong,resx,resy);
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
    gl.uniform2f(loc_res_main,resx,resy);
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
    gl.framebufferTexture2D(gl.FRAMEBUFFER, pongAttachmentPoint, gl.TEXTURE_2D, pongTex, 0);


 

   
//    while (!glyphLoaded) {
    
//    }
    requestAnimationFrame(drawScene);

    function drawScene(time) {

        time = (performance.now())* .001;

    
        //main canvas stuffs
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0,0,resx,resy);
        gl.clearColor(0, 0, 1, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.useProgram(mainProgram);
        gl.uniform1f(loc_time_main,time);
        gl.bindVertexArray(vao_main);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        firstframe = false;
        requestAnimationFrame(drawScene);
    }

    
}

main();