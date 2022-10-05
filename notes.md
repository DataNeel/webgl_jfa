https://www.shadertoy.com/results?query=tag=jfa
https://blog.demofox.org/2016/02/29/fast-voronoi-diagrams-and-distance-dield-textures-on-the-gpu-with-the-jump-flooding-algorithm/
https://shaderbits.com/blog/various-distance-field-generation-techniques
https://bgolus.medium.com/the-quest-for-very-wide-outlines-ba82ed442cd9
https://www.youtube.com/watch?v=A0pxY9QsgJE
https://www.youtube.com/watch?v=AT0jTugdi0M

https://github.com/cacheflowe/haxademic/blob/master/data/haxademic/shaders/filters/jump-flood.glsl
https://github.com/cacheflowe/haxademic/blob/master/src/com/haxademic/demo/draw/filters/shaders/Demo_JumpFlood_SDF.java


https://editor.p5js.org/yangshuzhan/sketches/SdzrD0Pqv


1. clear out shader (intentionally)
2. load texture into main shader
3. load texture into frame buffer
4. show frame buffer in main shader
5. make framebuffer 32 bit


6. process texture using jfa
7. render jfa in main texture
8. use jfa as distance map