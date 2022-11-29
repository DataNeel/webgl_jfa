//broken
//0x788746d8311589f8ba527c9b7aeb15579ca4c06871272ee7cd7dda29311fd79d
//0xa7ce56a0e65f548b7f98942a43d93b89cec00b6f4dd8da6b50fa2b1db818e5a5
// 0x887e5ec185b6b668a22926f00869eacd33bbad3f11e9b8136a0023aca3a12c5b

//keyboard shit
showBones = 0;
showDist = 0;
document.addEventListener('keydown', (event) => {
  var name = event.key;
  var code = event.code;
  // Alert the key name and key code on keydown
  if (['1','2','3'].includes(name)) {
    showBones = name == 2? 1:0;
    showDist = name == 3? 1:0;
  }
  
}, false);

const TWO_PI = 6.28318530717958647693;
function genTokenData(projectNum) {
    let data = {};
    let hash = "0x";
    for (var i = 0; i < 64; i++) {
      hash += Math.floor(Math.random() * 16).toString(16);
    }
    data.hash = hash;
    data.tokenId = (projectNum * 1000000 + Math.floor(Math.random() * 1000)).toString();
    return data;
  }
  let tokenData = genTokenData(109);
  // tokenData.hash = '0xfe7f3aba50202bee6b904bc7c6a51da31df8211263b117324233e736055443af';
  console.log(tokenData.hash);
  

  let xs=Uint32Array.from([0,0,0,0].map((_,i)=>parseInt(tokenData.hash.substr(i*8+2,8),16)));
  const R=()=>{let s,t=xs[3];xs[3]=xs[2];xs[2]=xs[1];xs[1]=s=xs[0];t^=t<<11;t^=t>>>8;xs[0]=t^s^(s>>>19);return xs[0]/0x100000000};
  const RN=(a,b)=>a+(b-a)*R();
  const RI=(a,b)=>~~RN(a,b+1);
  const RV=(v)=>v[RI(0,v.length-1)];
  

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

glyph_v = `#version 300 es
precision highp float;
in vec3 a;
out vec2 u;
out float l;
void main() {
  u = a.xy * 2. - 1.;
  l = a.z;
  gl_Position = vec4(u,0,1);
}`;

src_glyph = `#version 300 es
precision highp float;

in vec2 u;
in float l;
out vec4 cc;
uniform int c;

void main() {
  if (c == 0) {
    cc=vec4(l,0,0,1);
  }
  else if (c == 1) {
    cc=vec4(0,l,0,1);
  }
  else if (c == 2) {
    cc=vec4(l,l,0,1);
  }
  else {
    cc=vec4(0,0,0,1);
  }
}`;

// fragment shader for main canvas
src_f = `#version 300 es
precision highp float;

in vec2 u;
out vec4 cc;
uniform vec2 res;
uniform float time;
uniform sampler2D jfa;
uniform sampler2D bones;
uniform bool showBones;
uniform bool showDist;


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

float aastep(float threshold, float value) {
  #ifdef GL_OES_standard_derivatives
    float afwidth = length(vec2(dFdx(value), dFdy(value))) * .70710678118654757;
    return smoothstep(threshold-afwidth, threshold+afwidth, value);
  #else
    return step(threshold, value);
  #endif  
}

void main() {
    vec2 uv = gl_FragCoord.xy/res;
    vec4 t = texture(jfa,uv);
    vec4 boneT = texture(bones,uv);
    // uv.x *= res.x/res.y;
    float d1 = distance(vec2(t.x,t.y),uv)*2.;
    float d2 = distance(vec2(t.b,t.a),uv)*2.;
    //not sure, but makes a difference
    float d = .005;
    //.1 to .3
    float scale = .2;
    float n1 = d + snoise(vec3(uv*50.,time*.5))*d*scale;
    float n2 = d + snoise(vec3(uv*30.,-time*.5))*d*scale;
    
    //12 to 22 for the first number
    //0025 to 0005 for the last number
    float space = 15.*max(d1*d2,.0016);
    float width = space*.15;
 
    float dist = aastep(mod(d1,space),n1+width)-aastep(mod(d1,space),n1-width);
    float dist2 = aastep(mod(d2,space),n2+width)-aastep(mod(d2,space),n2-width);

    // float dist = step(d1,.002);
    // float dist2 = step(d2,.0075);

    vec4 c1 = vec4(59, 0, 134,255.)/255.;
    vec4 c2 = vec4(227, 208, 216,255.)/255.;
    vec4 c3 = vec4(180, 62, 143,255.)/255.;
    vec4 c4 = vec4(114, 225, 209,255.)/255.;
    
    vec4 ca = mix(c1,c2,dist);
    vec4 cb = mix(c3,c4,dist);
    cc = mix(ca,cb,dist2);

    if (showBones) {
      cc = boneT;
    }
    else if (showDist) {
      cc = t;
    }
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
    vec4 t = texelFetch(glyph,ivec2(gl_FragCoord.xy),0);
    if(t.r>0.)cc=vec4(uv,cc.ba);
    if(t.g>0.)cc=vec4(cc.rg,uv);
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

//draw a rock
function notsquare(x, y, w, sides) {
    let rads = [];
    for (let i = 0; i < sides; i++) {
        let r = R()*TWO_PI;
      rads.push(r);
      
    }
    rads.sort();
    let points = [];
    for (r of rads) {
        points.push([x+Math.cos(r)*w,y+Math.sin(r)*w]);
    }
    return points;
      }


function compareByAngle(coords, origin) {
    return function (a, b) {
        let coordxy = [coords.x,coords.y]
        acoords = [a.x,a.y];
        bcoords = [b.x,b.y];
        let c = [coordxy[0]-origin[0],coordxy[1]-origin[1]];
        adiff = [acoords[0]-coordxy[0],acoords[1]-coordxy[1]];
        bdiff = [bcoords[0]-coordxy[0],bcoords[1]-coordxy[1]];
        
        let dotA = adiff[0]*c[0]+adiff[1]*c[1];
        let detA = adiff[0]*c[1] - adiff[1]*c[0];
        let dotB = bdiff[0]*c[0]+bdiff[1]*c[1];
        let detB = bdiff[0]*c[1] - bdiff[1]*c[0];
        aangle = Math.atan2(detA,dotA);
        bangle = Math.atan2(detB,dotB);
        if (aangle < bangle) return -1;
        if (aangle > bangle) return 1;
        return 0;
    };
}

//crawler
class crawler {
 
    constructor(start) {
      this.currentNode = start;
      this.currentNode.taken = true;
      this.nodeHistory = [this.currentNode];
      this.pathNodes = [];
      this.crawlPosition = 0;
      this.paths = [];
      this.allPath = [[this.currentNode.x,this.currentNode.y]];
    }
    
    newNode(node) {
     this.currentNode = node; 
    }
    crawl() {
      let sortedNeighbors = []; //this.currentNode.neighbors.slice(this.currentNode.neighborsCrawled);
      for (let i of this.currentNode.neighbors) {
        if (!i.taken) {
          sortedNeighbors.push(i);
        }
      }
   
        sortedNeighbors.sort(compareByAngle(this.currentNode, this.allPath[this.allPath.length-Math.min(2,this.allPath.length)]));
      
       var neighbor = sortedNeighbors[0];
      
       if (neighbor.taken == false) {
        neighbor.taken = true;
        if (this.pathNodes.length == 0) {
         if (this.crawlPosition > 0) {
           this.pathNodes.push(this.nodeHistory[this.crawlPosition - 1]);
         }
         this.pathNodes.push(this.currentNode);
        }
        this.pathNodes.push(neighbor);
         this.allPath.push([neighbor.x,neighbor.y]);
        this.nodeHistory.push(neighbor);
        this.currentNode = neighbor;
        this.crawlPosition++;
       }
    }
    
    backTrack() {
      if (this.pathNodes.length > 0) {
        this.allPath.push([this.currentNode.x,this.currentNode.y]);
       this.stashPath();
      }
      
     this.nodeHistory.splice(this.crawlPosition,1);
     this.crawlPosition--;
     var backNode = this.nodeHistory[this.crawlPosition];
      this.currentNode = backNode;  
      this.allPath.push([backNode.x,backNode.y]);
     if (backNode.spent() == false) {
         
       
     }
    }
    
    stashPath() {
      var path = [];
      for (let i = this.pathNodes.length - 1;  i >= 0; i--) {
        var pathNode = this.pathNodes[i];
        path.push([pathNode.x, pathNode.y]);
        this.pathNodes.splice(i,1);
       }
       this.paths.push(path);
    }
    
    canCrawl() {
     return !this.currentNode.spent(); 
    }
    
    canBackTrack() {
     return(this.crawlPosition > 0); 
    }
    
   graphSpent() {
     return((this.canCrawl() == false) && (this.canBackTrack() == false)); 
    }
    
    keepCrawling() {
      while (this.graphSpent() == false) {
       if (this.canCrawl()) {
        this.crawl();
       }
       else if (this.canBackTrack()) {
        this.backTrack(); 
       }
      }
      this.allPath.pop();
      // print(c1.allPath[0],c1.allPath[c1.allPath.length-1]);
    }
   }

   function lerp(a, b, t) {
    return (1 - t) * a + t * b;
  }

   function chaikinPath(points, iterations, placement) {
    var smoothedPoints = points;
     for (let s = 0; s < iterations; s++) {
       var chaikinPoints = [];
       for (let i = 0; i < smoothedPoints.length; i++) {
         let point1 = smoothedPoints[i];
         let point2 = smoothedPoints[(i+1)%smoothedPoints.length];
         for (let j = 0; j < placement.length; j++) {
           let cp0 = lerp(point1[0],point2[0],placement[j]);
           let cp1 = lerp(point1[1],point2[1],placement[j]);
           chaikinPoints[2 * i + j] = [cp0, cp1];
           
        } 
       }
       smoothedPoints = chaikinPoints;
     }
   return smoothedPoints;
   
  }   

  function vsub(i, j) {
    return [j[0]-i[0],j[1]-i[1]];
  }
  function vadd(i, j) {
    return [j[0]+i[0],j[1]+i[1]];
  }

  function vnorm(i) {
    let c = Math.sqrt(i[0]**2+i[1]**2);
    if (c > 0) {
        return [i[0]/c,i[1]/c];
    }
    else {
        return [0,0];
    }
  }

  function vrotate(v, r) {
    return [v[1],-1*v[0]];
    // let [x,y] = v;
    // return [x*Math.cos(r)-y*Math.sin(r),x*Math.sin(r)+y*Math.cos(r)];
  }


  const calculateNormals = (p) => {
    let edges = [];
    for (let i = 0; i < p.length; i++) {
      let edge = vsub(p[(i+1)%p.length], p[i]);
      edge = vnorm(edge);
      edge = vrotate(edge, TWO_PI/4);
      edges.push(edge);
    }
  
    let VOs = [];
  
    for (let i = 0; i < edges.length; i++) {
      let VO = vadd(edges[i], edges[(i + 1)%edges.length]);
      VO = vnorm(VO);
    //   p[(i+1)%p.length].normal = VO;
      VOs.push(VO);
    }
    return VOs;
  }

  function lengthOnPath(p) {
    let l = [];
    l.push(0);
    for (let i = 1; i < p.length; i++) {
      let d = Math.sqrt((p[i][0]-p[i-1][0])**2 + (p[i][1]-p[i-1][1])**2);
      l.push(l[i-1]+d);
    }
    ml = l[l.length-1];
    for (i in l) {
      l[i] /=ml;
    }
    return l;
  }

  function flatWithLength(p) {
    let l = lengthOnPath(p);
    let newP = [];
    for (i in p) {
      newP.push(p[i][0]);
      newP.push(p[i][1]);
      newP.push(l[i]);
    }
    return newP;
  }
  
  

  
  //vector mult is p5
  //vector add is p5
  const offset = (vs,ns,o) => {
    let ov = [];
    for (i in vs) {
      let v = [ns[i][0]*o,ns[i][1]*o];
      let vo = vadd(vs[i],v);
      
      ov.push(vo)
    }
    return ov; 
    
  }
  


function main() {
    let paths = [];
    pebbles = [];
    let nodew = RI(4,10);
    let nodeh = RI(4,10);
    let order1 = R()>.5;
    console.log(nodew, nodeh);
    for (form = 0; form < 2; form++) {
        xs=Uint32Array.from([0,0,0,0].map((_,i)=>parseInt(tokenData.hash.substr(i*8+2,8),16)));
        nodes = [];
        for (index = 0; index < nodew * nodeh; index++) {
            let [i, j] = order1 ? [index % nodew, index / nodew | 0] : [index / nodeh | 0, index % nodeh]
            let x = lerp(.05,.95, (i + .5) / nodew) + (.5-R()) * .05;
            let y = lerp(.05,.95, (j + .5) / nodeh) + (.5-R()) * .05;
            nodes.push({
            x: x,
            y: y,
            i: i,
            j: j,
            r: R(),
            taken: false,
            neighborsCrawled: 0,
            neighbors: [],
            spent: function () {
                let freeNeighbors = [];
                for (let i of this.neighbors) {
                if (!i.taken) {
                    freeNeighbors.push(i);
                }
                }
                return freeNeighbors.length==0;
            }
            });
        
        }
        for (let i = 0; i < nodes.length*.1; i++) {
          let pi = RI(0,nodes.length-1);
          let p = nodes.splice(pi,1);
          if (R()>.25) {
            pebbles.push(p);
          }
          
        }
        
        for (let i = 0; i < nodes.length - 1; i++) {
            rules =  [1,
                        RI(0,1),
                        RI(0,1),
                        1];
        
        for (let j = i + 1; j < nodes.length; j++) {
        
            
            var b1 = Boolean(Math.abs(nodes[i].i - nodes[j].i) == rules[0]) && Boolean(Math.abs(nodes[i].j - nodes[j].j) == rules[1] ) ;
            var b2 = Boolean(Math.abs(nodes[i].j - nodes[j].j) == rules[2]) && Boolean(Math.abs(nodes[i].i - nodes[j].i) == rules[3]) ;
            // let dropProb = lerp(maxDropProb,minDropProb,(lifeform)/(forms))
            let dropProb = .8 - .1 * form;
            if (Boolean(b1 || b2) && Math.abs(nodes[i].r-nodes[j].r) < dropProb) {
            nodes[i].neighbors.push(nodes[j]);
            nodes[j].neighbors.push(nodes[i]);
            }
        }
        }
        let c1 = new crawler(RV(nodes));
            c1.keepCrawling();
            // console.log(c1);
            c1.allPath.push(c1.allPath[0]);
            let ap = chaikinPath(c1.allPath,3,[.1,.9])
            let norms = calculateNormals(ap);
            let vo = offset(ap,norms,.005 - form*.0045);
          
            paths.push(vo);
    }




















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
    dpr = devicePixelRatio * 2;
    let minRes = Math.min(w, h);
    h = w =  minRes * 1.;
    // w = h * 16/9;
    resx = C.width = w * dpr | 0;
    resy = C.height = h * dpr | 0;
    // resx, resy = 1024;
    C.style.width = w + 'px';
    C.style.height = h + 'px';

    let positions = Float32Array.of(0, 1, 0, 0, 1, 1, 1, 0);
    
    

    // create glyph framebuffer
    const glyph = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, glyph);
    const glyphAttachmentPoint = gl.COLOR_ATTACHMENT0;

    //create glyph texture
    var glyphTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + 0);
    gl.bindTexture(gl.TEXTURE_2D, glyphTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, resx, resy, 0, gl.RGBA, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.framebufferTexture2D(gl.FRAMEBUFFER, glyphAttachmentPoint, gl.TEXTURE_2D, glyphTex, 0);

    //glyph program
        //set up the ping program
    glyphProgram = createProgram(glyph_v, src_glyph);
    gl.useProgram(glyphProgram)
    loc_a_glyph = gl.getAttribLocation(glyphProgram, 'a');
    loc_b_glyph = gl.getAttribLocation(glyphProgram, 'b');
    loc_c_glyph = gl.getUniformLocation(glyphProgram, 'c');
    
    //create a buffer and vao for glyph
    positionBuffer_glyph = gl.createBuffer();
    vao_glyph = gl.createVertexArray();
    gl.bindVertexArray(vao_glyph);
    gl.enableVertexAttribArray(loc_a_glyph);
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer_glyph);
    gl.vertexAttribPointer(loc_a_glyph, 3, gl.FLOAT, false, 0, 0);

    //draw glyph
    gl.bindFramebuffer(gl.FRAMEBUFFER,glyph);
    gl.viewport(0,0,resx,resy);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(glyphProgram);
    gl.bindVertexArray(vao_glyph);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    let glyphpositions =    new Float32Array(flatWithLength(paths[0]));
    gl.uniform1i(loc_c_glyph,0);
    gl.bufferData(gl.ARRAY_BUFFER, glyphpositions, gl.STATIC_DRAW);
    gl.drawArrays(gl.LINE_LOOP, 0, glyphpositions.length/3.);
    glyphpositions =    new Float32Array(flatWithLength(paths[1]));
    gl.uniform1i(loc_c_glyph,1);
    gl.bufferData(gl.ARRAY_BUFFER, glyphpositions, gl.STATIC_DRAW);
    gl.drawArrays(gl.LINE_LOOP, 0, glyphpositions.length/3.);
    gl.uniform1i(loc_c_glyph,2);
    for (let i = 0; i < pebbles.length/2; i++) {
      p = pebbles[i][0];
      let pts = new Float32Array(flatWithLength(notsquare(p.x,p.y,.02,RI(3,6))));
      gl.bufferData(gl.ARRAY_BUFFER, pts, gl.STATIC_DRAW);
      gl.drawArrays(gl.LINE_LOOP, 0, pts.length/3.);

    }
    gl.disable(gl.BLEND);
    


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

    //create a buffer and vao for pong
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
    loc_bones_main = gl.getUniformLocation(mainProgram, 'bones');
    loc_showBones_main = gl.getUniformLocation(mainProgram, 'showBones');
    loc_showDist_main = gl.getUniformLocation(mainProgram, 'showDist');
    loc_a_main = gl.getAttribLocation(mainProgram, 'a');
    // //debug change
    gl.uniform1i(loc_buffer_main,1);
    gl.uniform1i(loc_bones_main,0);
    

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


        //glyph texture
        gl.activeTexture(gl.TEXTURE0+0);
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
        let steps = Math.ceil(Math.log2(resx));
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
            gl.uniform1f(loc_step_ping,step);
            gl.uniform1f(loc_frame_ping,frame);
            gl.bindVertexArray(vao_ping);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

					frame++
        }


    
 

   

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
        gl.uniform1i(loc_showBones_main,showBones);
        gl.uniform1i(loc_showDist_main,showDist);
        gl.bindVertexArray(vao_main);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        firstframe = false;
        requestAnimationFrame(drawScene);
    }

    
}

main();