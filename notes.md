https://www.shadertoy.com/results?query=tag=jfa

https://blog.demofox.org/2016/02/29/fast-voronoi-diagrams-and-distance-dield-textures-on-the-gpu-with-the-jump-flooding-algorithm/

https://shaderbits.com/blog/various-distance-field-generation-techniques

https://bgolus.medium.com/the-quest-for-very-wide-outlines-ba82ed442cd9

https://www.youtube.com/watch?v=A0pxY9QsgJE

https://www.youtube.com/watch?v=AT0jTugdi0M



https://github.com/cacheflowe/haxademic/blob/master/data/haxademic/shaders/filters/jump-flood.glsl
https://github.com/cacheflowe/haxademic/blob/master/src/com/haxademic/demo/draw/filters/shaders/Demo_JumpFlood_SDF.java


https://editor.p5js.org/yangshuzhan/sketches/SdzrD0Pqv




* combining the code that makes the structure with the code that makes the texture (right now, this is actually the output of two pieces of code)
    * try drawing in webgl
        start with circle
        then port code
            remove p5 vector
            use code to create arrays of points
            draw lines instead of circle
            figure out blend mode
                https://www.html5gamedevs.com/topic/19785-blendmode-add-in-webgl/
            draw both glyphs
    * try drawing in js and passing to webgl
    * try drawing in p5 and passing to webgl

* adapting to a different aspect ratio (involves handling dimensions in the shader)

* smoothing out some edges (they’re literally bumpy, and I have some hypotheses as to why)

* Handling more variables (dimensions, width, density, etc)

* adding more negative space (see above)

* adding slight texture in the negative space