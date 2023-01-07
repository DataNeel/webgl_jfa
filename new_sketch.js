let c1;
//broken

//keyboard shit
showBones = 0;
showDist = 0;
showGradient = 0;
document.addEventListener('keydown', (event) => {
  var name = event.key;
  var code = event.code;
  // Alert the key name and key code on keydown
  if (['1','2','3','4'].includes(name)) {
    showBones = name == 2? 1:0;
    showDist = name == 3? 1:0;
    showGradient = name == 4? 1:0;
    console.log(name);
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
  // tokenData.hash = '0xd5b83154add2532e9aa73269b5c232a02ed4ee0d131125682baec16cd306049e';
  console.log(tokenData.hash);
  

  let xs=Uint32Array.from([0,0,0,0].map((_,i)=>parseInt(tokenData.hash.substr(i*8+2,8),16)));
  const R=()=>{let s,t=xs[3];xs[3]=xs[2];xs[2]=xs[1];xs[1]=s=xs[0];t^=t<<11;t^=t>>>8;xs[0]=t^s^(s>>>19);return xs[0]/0x100000000};
  const RN=(a,b)=>a+(b-a)*R();
  const RI=(a,b)=>~~RN(a,b+1);
  const RV=(v)=>v[RI(0,v.length-1)];


let palettes = [
  [[11, 10, 7],[44, 87, 132],[86, 136, 199],[208, 244, 234]] //hyperlink
  ,[[22,15,41],[36,106,115],[54,143,139],[243,223,193]] //steel/linen
  ,[[8, 76, 97],[219, 80, 74],[255, 208, 70],[61, 165, 217]] //primary
  ,[[0, 48, 73],[214, 40, 40],[247, 127, 0],[252, 191, 73]] //red and orange
  ,[[0, 56, 68],[0, 108, 103],[241, 148, 180],[255, 177, 0]] //tropical
  ,[[30, 33, 43],[77, 139, 49],[255, 200, 0],[255, 132, 3]] //green and yellow
  ,[[38, 38, 38],[158, 158, 158],[42, 114, 33],[237, 237, 237]] //green
  ,[[0,	43,	54],[42,	161,	152],[211,	54,	130],[181,	137,	0]] //solarized
];

let palette = RV(palettes);

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
uniform bool showGradient;




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
  float pi = 3.14159265359;
    vec4 t = texelFetch(jfa,ivec2(gl_FragCoord.xy),0);
    vec4 boneT = texelFetch(bones,ivec2(gl_FragCoord.xy),0);
    float boneR = texelFetch(bones,ivec2(t.rg),0).r;
    float boneG = texelFetch(bones,ivec2(t.ba),0).g;

    float dscale = .1;
    float d1 = dscale*length(vec2(t.x,t.y)-gl_FragCoord.xy)/max(res.x,res.y);
    float d2 = dscale*length(vec2(t.b,t.a)-gl_FragCoord.xy)/max(res.x,res.y);
    float n1 = -.1+snoise(vec3(d1*100.,boneR*200.,time*.05));
    float n2 = -.1+snoise(vec3(d2*10.,boneG*20.,time*.05));


  float dist = pow(step(clamp(sin(boneR*20.+time/50.),0.,1.),n1),2.);
  float dist2 = pow(clamp(sin(boneG*10.+time/32.)+.0,0.,1.),5.);

    vec4 c1 = vec4(${palette[0].toString()},255)/255.;
    vec4 c2 = vec4(${palette[1].toString()},255.)/255.;
    vec4 c3 = vec4(${palette[2].toString()},255.)/255.;
    vec4 c4 = vec4(${palette[3].toString()},255.)/255.;

    vec4 ca = mix(c1,c2,dist);
    vec4 cb = mix(c3,c4,dist);
    cc = mix(ca,cb,dist2);

    if (showBones) {
      cc = vec4(step(0.00000001,boneT.r),step(0.000000000001,boneT.g),step(0.000000000001,boneT.b),1.);
    }
    if (showGradient) {
      cc = vec4(t.x,t.y,res.x,res.x)/res.x;
    }
    if (showDist) {
      cc = vec4(sin(boneR*20.),1.,1.,1.);
    }
    
  
}`;


//fragment shader for init
src_init = `#version 300 es
precision highp float;

in vec2 u;
out vec4 cc;
uniform sampler2D glyph;

void main() {
    cc=vec4(-100000000000.);
    vec4 t = texelFetch(glyph,ivec2(gl_FragCoord.xy),0);
    if(t.r>0.0)cc=vec4(gl_FragCoord.xy,cc.ba);
    if(t.g>0.0)cc=vec4(cc.rg,gl_FragCoord.xy);
}`;

//fragment shader for ping
src_ping = `#version 300 es
precision highp float;
in vec2 u;
out vec4 cc;
uniform vec2 res;
uniform sampler2D pong;
uniform float frame;

void main() {
    
   ivec2 offset = ivec2(max(res.x, res.y) * .5 / pow(2.,frame));
   vec4 t = texelFetch(pong,ivec2(gl_FragCoord.xy),0);
   float bestDistR = length(t.xy-gl_FragCoord.xy);
   float bestDistG = length(t.ba-gl_FragCoord.xy);
   for (float i = -1.; i <=1.; i++) {
       for (float j = -1.; j <=1.; j++) {
          ivec2 pix = ivec2(gl_FragCoord.xy)+ivec2(i,j)*offset;
          vec4 t2 = texelFetch(pong,pix,0);
          float new_dist = 0.;
          if (t2.x > 0.) {
           new_dist = length(t2.xy-gl_FragCoord.xy);
           if (new_dist<bestDistR) {
               t.rg = t2.rg;
               bestDistR = new_dist;
           }
          }
          if (t2.b > 0.) {
           new_dist = length(t2.ba - gl_FragCoord.xy);
           if (new_dist<bestDistG) {
                t.ba = t2.ba;
                bestDistG = new_dist;
           }
          }
       }
   }
	cc=t;;
}`;

//fragment shader for pong
src_pong = `#version 300 es
precision highp float;

out vec4 cc;
uniform sampler2D ping;

void main() {
    cc = texelFetch(ping,ivec2(gl_FragCoord.xy),0);
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
      ///
      // l[i] /=ml;
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
    let nodeh = RI(3,10);
    let nodew = RI(5,12);
    let order1 = R()>.5;
    console.log(nodew, nodeh);
    for (form = 0; form < 2; form++) {
        xs=Uint32Array.from([0,0,0,0].map((_,i)=>parseInt(tokenData.hash.substr(i*8+2,8),16)));
        nodes = [];
        for (index = 0; index < nodew * nodeh; index++) {
            let [i, j] = order1 ? [index % nodew, index / nodew | 0] : [index / nodeh | 0, index % nodeh]
            let nodebuffer = -.05;
            let x = lerp(0+nodebuffer,1-nodebuffer, (i + .5) / nodew) + (.5-R()) * .05;
            let y = lerp(0+nodebuffer,1-nodebuffer, (j + .5) / nodeh) + (.5-R()) * .05;
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
          // if (R()>.25) {
            // pebbles.push(p);
          // }
          
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
            let dropProb = 1.- .3 * form;
            if (Boolean(b1 || b2) && Math.abs(nodes[i].r-nodes[j].r) < dropProb) {
            nodes[i].neighbors.push(nodes[j]);
            nodes[j].neighbors.push(nodes[i]);
            }
        }
        }
        c1 = new crawler(RV(nodes));
            c1.keepCrawling();
            console.log(c1);
            console.log(c1.allPath.length);
            // c1.allPath.push(c1.allPath[0]);
            console.log(c1.allPath.length);
            let ap = chaikinPath(c1.allPath,10,[.1,.9])
            ap.push(ap[0]);
            let norms = calculateNormals(ap);
            let vo = offset(ap,norms,.000005);
          
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
    dpr = devicePixelRatio *2;
    let minRes = h;//Math.min(w, h);
    h = w =  minRes * 1.;
    //  h = w * 9/16;
    w = h *16/9;
    // w = h * 4/5;
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
    gl.lineWidth(.1);
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
    loc_showGradient_main = gl.getUniformLocation(mainProgram, 'showGradient');
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
        var steps = Math.ceil(Math.log2(Math.max(resx,resy)));
        console.log(steps);
			let frame=0
        for (let i = -2; i <steps+2; i++) {
            //pong
            let frame = i;
            if (i < 0) {
              frame = steps-1;
            }
            
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
            gl.uniform1f(loc_frame_ping,Math.min(frame,steps));
            gl.bindVertexArray(vao_ping);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);


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
        gl.uniform1i(loc_showGradient_main,showGradient);
        gl.bindVertexArray(vao_main);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        firstframe = false;
        requestAnimationFrame(drawScene);
    }

    
}

main();