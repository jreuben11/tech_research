### Interactive Terminals

#### 'qt' Terminal
The 'qt' terminal is a modern, multiplatform interactive driver that leverages the Qt library (Qt4 or Qt5) for high-quality 2D painting and rendering. It supports interactive features like zooming, panning, and mouse input, as well as menu-driven export to formats like PNG, SVG, or PDF without replotting. The terminal is written in C++, as Qt integration is part of gnuplot's modular subsystems. It is configured as the default interactive terminal if Qt is detected during build, and it is noted for being the fastest and most feature-rich option among interactive terminals. Font initialization can be slow on some systems like macOS, and it relies on shared system font caches.

Technical details include:
- Rendering is handled through Qt's QPainter class for 2D graphics, which supports antialiasing, transformations, and compositing.
- Interactive elements are managed via QWidget or derived classes for window creation and event handling (e.g., mouse events for coordinate readout or hotkeys).
- Output export uses Qt's built-in capabilities for vector and raster formats.

Source file: The implementation is in `term/qt.trm` within the gnuplot source code repository (available at https://sourceforge.net/p/gnuplot/gnuplot-main/ci/master/tree/term/qt.trm). This file defines the terminal's entry point, options, and function table for operations like initialization (`QT_init`), graphics setup (`QT_graphics`), line drawing (`QT_linetype`), and text rendering (`QT_text`).

No specific code snippets are available from searches, but typical structure in .trm files includes macros like `TERM_PUBLIC void QT_init()` for setting up the Qt application and canvas, and drawing functions using `QPainter::drawLine` or `QPainter::fillPath` for paths.

#### 'wxt' Terminal
The 'wxt' terminal uses wxWidgets for the GUI framework, combined with the Cairo library for 2D vector rendering and Pango for text layout and rendering. It is multiplatform (Linux, Windows, macOS) and supports interactive display with features like mouse input and coordinate printing. Rendering is primarily software-based on GTK (using Cairo's image surface), which can be slow for complex plots due to frequent stroking operations. On Windows, it uses native GDI for better performance and font quality. Hardware acceleration via X11 Render extension or OpenGL (glitz) is possible but not default or well-supported in gnuplot's implementation. The terminal requires libcairo, libpango, and libwxgtk during build.

Technical details include:
- Cairo handles drawing primitives (lines, polygons, fills) on surfaces like image or GTK surfaces.
- Pango is used for font rendering, supporting UTF-8 and complex layouts.
- wxWidgets manages the window, events, and integration with Cairo contexts.

Source file: The core is in `term/wxt.trm`, with supporting code in `src/wxterminal/wxt_gui.cpp` (https://sourceforge.net/p/gnuplot/gnuplot-main/ci/master/tree/term/wxt.trm). The .trm file sets up the terminal table, while wxt_gui.cpp contains the Cairo integration.

Code snippet example (from discussions on Cairo integration in patches):
```
#if defined(IMAGE_SURFACE)
  cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
#else // GTK_SURFACE
  cairo_surface_t *surface = cairo_xlib_surface_create(display, drawable, visual, width, height);
#endif
cairo_t *cr = cairo_create(surface);
// Drawing operations...
cairo_stroke(cr);
```
This shows surface creation for software (image) or X11 rendering, and stroking paths.

#### 'x11' Terminal
The 'x11' terminal is the classic interactive driver, using Xlib directly for drawing on the X Window System. It is suitable for Unix-like systems and supports basic interactivity like mouse input and window resizing, but lacks the advanced features and quality of newer terminals like qt or wxt. It is built by default and can be disabled via configure flags. Resizing may trigger replots, but this can be suppressed with `set term x11 noreplotonresize`. It supports multibyte fonts via environment variables like `mbfont`.

Technical details include:
- Direct use of Xlib functions for window creation, event polling, and primitive drawing (points, lines, text).
- No external libraries like Cairo; rendering is software-based and tied to X Server capabilities.
- Supports color management via XColor and GC (Graphics Context).

Source file: Implemented in `term/x11.trm` (https://sourceforge.net/p/gnuplot/gnuplot-main/ci/master/tree/term/x11.trm), with additional code in `src/x11.gih`.

No direct snippets found, but typical Xlib usage includes:
```
Display *display = XOpenDisplay(NULL);
Window win = XCreateSimpleWindow(display, RootWindow(display, screen), x, y, width, height, ...);
GC gc = XCreateGC(display, win, 0, NULL);
XDrawLine(display, win, gc, x1, y1, x2, y2);
```
This illustrates window setup and line drawing.

### Non-Interactive Terminals

These are primarily software-based 2D renderers focused on file output, without interactive windows.

#### 'pdfcairo'/'pngcairo' (Cairo-based)
These terminals use the Cairo library for vector and raster rendering, with Pango for text. 'pdfcairo' outputs to PDF, while 'pngcairo' outputs to PNG. They are recommended over older terminals for better quality, UTF-8 support, and antialiasing. Built if libcairo and libpango are present; they support enhanced text and transparent backgrounds. 'pdfcairo' replaces the deprecated PDFlib-based PDF terminal.

Technical details include:
- Cairo creates surfaces (e.g., PDF or image) for path-based drawing, filling, and stroking.
- Pango handles text layout for complex fonts.
- Output is finalized by writing the surface to file.

Source files: Base Cairo logic in `term/cairo.trm`, PDF-specific in `term/pdf.trm`, PNG in `term/png.trm` (Cairo variant selected at configure time) (e.g., https://sourceforge.net/p/gnuplot/gnuplot-main/ci/master/tree/term/cairo.trm).

Code snippet example (from Cairo discussions):
```
cairo_surface_t *surface = cairo_pdf_surface_create(filename, width, height);
cairo_t *cr = cairo_create(surface);
cairo_move_to(cr, x, y);
cairo_line_to(cr, x2, y2);
cairo_stroke(cr);
cairo_surface_destroy(surface);
```
This demonstrates PDF surface setup and path drawing.

#### 'svg' Terminal
The 'svg' terminal generates Scalable Vector Graphics files, suitable for web or vector editing. It supports embedding in terminals like domterm and allows interactivity (zooming, toggling) when viewed in browsers. It handles paths, transformations, and text as SVG elements.

Technical details include:
- Output is XML-based, with elements for lines, polygons, text, and groups.
- Supports colors, dashes, and fonts via CSS or inline styles.

Source file: `term/svg.trm` (https://sourceforge.net/p/gnuplot/gnuplot-main/ci/master/tree/term/svg.trm).

No snippets found, but typical output generation involves fprintf to file with tags like `<path d="M x y L x2 y2" stroke="color"/>`.

#### 'postscript' Terminal
The 'postscript' terminal produces PostScript output for printing or conversion to EPS/PDF. It supports enhanced text, colors, and fonts, but is vector-based without external libraries.

Technical details include:
- Generates PostScript commands for page setup, drawing, and fonts.
- Handles Level 1/2/3 PostScript features.

Source file: `term/post.trm` (https://sourceforge.net/p/gnuplot/gnuplot-main/ci/master/tree/term/post.trm).

Snippet example (generic PostScript generation):
```
fprintf(outfile, "/M {moveto} bind def\n");
fprintf(outfile, "%d %d M\n", x, y);
fprintf(outfile, "%d %d L stroke\n", x2, y2);
```
This shows defining shortcuts and drawing lines.

#### 'dumb' Terminal
The 'dumb' terminal renders plots as ASCII art in a text block, using characters like '*' or '-' for points and lines. It is useful for debugging, console output, or when no graphics are available. Size is set in characters (default 80x25).

Technical details include:
- Uses a grid buffer to place characters representing plot elements.
- Simple algorithms for line rasterization (e.g., Bresenham-like) in text.

Source file: `term/dumb.trm` (https://sourceforge.net/p/gnuplot/gnuplot-main/ci/master/tree/term/dumb.trm).

No snippets, but it involves 2D array filling, e.g.:
```
char plot[WIDTH][HEIGHT];
plot[y][x] = '*';
for (int i = 0; i < HEIGHT; i++) printf("%s\n", plot[i]);
```
This represents grid-based rendering.

All terminals are primarily 2D software renderers, with no native OpenGL or Vulkan support. For full source, download gnuplot from https://sourceforge.net/projects/gnuplot/. Build process concatenates .trm files into term.c for the terminal driver table.