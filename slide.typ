#import "@preview/touying:0.4.2": *

#let s = themes.simple.register(aspect-ratio: "16-9")
#let (init, slides) = utils.methods(s)
#show: init

#let (slide, empty-slide) = utils.slides(s)
#show: slides

= Title

== First Slide

Hello, Touying!

#pause

Hello, Typst!

