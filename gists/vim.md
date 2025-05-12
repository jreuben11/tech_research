# keys

- $ vimtutor
- $ nvim
- .vimrc file
- `:` command mode `esc` normal mode (for navigating) `i` insert mode before cursor `a` insert mode after cursor `v` visual mode (for selecting). `A` insert mode from end of line `R` replace mode
- `h j k l` move left down up right `10l` move 10 chars right
- `gg` goto top `G` goto bottom
- `^`, `0` goto beginning of line `$` goto end of line
- `gj` move by visual line instead of logical line
- `0` goto beginning of line
- `5G` `:5` goto line 5
- `x` delete char `2x` delete 2 chars
- `dw` delete word until start of next `de` delete word until until end of word `d$` delete to end of line
- `dd` delete line `10dd` delete 10 lines
- d   number   motion - eg `d2w` delete 2 words
- `.` repeat last command `@:` repeat last Ex command `&` repeat `:substitute`command
- `u` undo. `Ctrl r` redo `U` undo all changes on line(DIDNT WORK) `Ctrl u` undo in edit mode.

- `y` yank (copy)
- `p` paste
- `r` replace current char
- `s` change char - delete current char and enter insert mode
- `S` change line - delete line and enter insert mode
- `ce` change until end of word - delete until end of word and enter insert mode
- `c$` change rest of line - delete until end of line and enter insert mode
- `o` - create new line and enter insert mode on next line  `O` - create new line and enter insert mode on previous line

- `w` goto 1st char of next word `3w` of 3 words forward
- `e` goto last char of next word
- `b` goto 1st char of preceding word
- `W` `E` `B` - as above, but ignore commas and parenthesis
- `fx` go forward until next char x `;` / `,` go forward / back
- `Fx` go backward until next char x
- `tx` go forward until next char preceding x
- `Tx` go backward until next char preceding x

- `/` search `n` next `N` previous `\c` ignore case for a search
- `:set ic` set ignore case for search `:set noic` disable
- `:set hlsearch` highlight all matching phrases `:set nohlsearch`
- `:set incsearch`       show partial matches for a search phrase
- `Ctrl-O` goto previous position `Ctrl-I` goto next position
- `%` goto matching parenthesis
- `:s/old/new/g`  substitute 'new' for 'old' globally in line, `:#,#s/old/new/g` between line numbers, `:%s/old/new/gc` percentage is globally in file, `c` is prompt on each
- `*` search for next occurence of word under cursor
- `:!ls` execute external command
- `:w filename` write to file
- `:r filename` retrieve contents of file and paste in
- `:e filename` open a file for editing eg `:e ~/.vimrc`
- `Ctrl-D` command completion
- `Ctrl-W Ctrl-W` switch window - eg to help window
- 
- `:help nvim` tutor
- `:checkhealth` to optimize
- `:help`
- `:x` other way to exit
- `:set number` display line numbers
- 
- `Ctrl k` , `Ctrl j` - zoom in / out [DIDNT WORK]
- 
- `vim -u NONE -N` - `-u NONE` dont source `~/.vimrc` on startup `-N` dont revert to vi compat mode
- `>G` increace indentation

# Vim cheatsheet

## Global

- :help keyword – open help for keyword
- `:o` file – open file
- `:saveas file` – save file as
- `:close` – close current window

## Cursor Movements
- 
- h – move cursor left
- j – move cursor down
- k – move cursor up
- l – move cursor right
- `H` – move to top of screen
- `M` – move to middle of screen
- `L` – move to bottom of screen
- w – jump forwards to the **start** of a word
- W – jump forwards to the start of a word (words can contain punctuation)
- `e` – jump forwards to the **end** of a word
- E – jump forwards to the end of a word (words can contain punctuation)
- b – jump backwards to the start of a word
- B – jump backwards to the start of a word (words can contain punctuation)
- 0 – jump to the start of the line
- ^ – jump to the first non-blank character of the line
- `$` – jump to the end of the line
- `g_` – jump to the last non-blank character of the line
- gg – go to the first line of the document
- G  – go to the last line of the document
- 
- `5G` – go to line 5
- f{char} – jump to next occurrence of character x
- t{char} – jump to before next occurrence of character x
- `}` – jump to next paragraph (or function/block, when editing code)
- `{` – jump to previous paragraph (or function/block, when editing code)
- zz – center cursor on screen
- Ctrl + b – move back one full screen
- Ctrl + f – move forward one full screen
- `Ctrl + d` – move forward 1/2 a screen
- `Ctrl + u` – move back 1/2 a screen

- Tip: Prefix a cursor movement command with a number to repeat it. For example, 4j moves down 4 lines.

## Insert Mode

- i – insert before the cursor
- I – insert at the beginning of the line
- a – insert (append) after the cursor
- A – insert (append) at the end of the line
- o – append (open) a new line below the current line
- O – append (open) a new line above the current line
- `ea` – insert (append) at the end of the word
- Esc – exit insert mode

## Editing

- r – replace a single character
- J – join line below to the current line
- cc – change (replace) entire line
- cw – change (replace) to the end of the word
- c$ – change (replace) to the end of the line
- s – delete character and substitute text
- S – delete line and substitute text (same as cc)
- xp – transpose two letters (delete and paste)
- u – undo
- Ctrl + r – redo
- . – repeat last command

## Marking Text (Visual Mode)

- v – start visual mode, mark lines, then perform an operation (such as d-delete)
- V – start linewise visual mode
- Ctrl + v – start blockwise visual mode
- o – move to the other end of marked area
- O – move to other corner of block
- aw – mark a word
- ab – a block with ()
- aB – a block with {}
- ib – inner block with ()
- iB – inner block with {}
- Esc – exit visual mode
- Visual Mode Commands
- > – shift text right
- < – shift text left
- y – yank (copy) marked text
- d – delete marked text
- `~` – switch case
- `viwS{char}` - surround selection with char (**doesnt support brackets**)

## Registers

- :reg – show registers content
- `"{char}y` – yank into register x
- `"{char}p` – paste contents of register x
- Tip: Registers are being stored in `~/.viminfo`, and will be loaded again on next restart of vim.
- Tip: Register 0 contains always the value of the last yank command.

## Marks

- :marks – list of marks
- ma – set current position for mark A
- `a – jump to position of mark A
- y`a – yank text to position of mark A

## Macros

- qa – record macro a
- q – stop recording macro
- @a – run macro a
- @@ – rerun last run macro

## Cut and Paste

- yy – yank (copy) a line
- 2yy – yank (copy) 2 lines
- yw – yank (copy) the characters of the word from the cursor position to the start of the next word
- y$ – yank (copy) to end of line
- p – put (paste) the clipboard after cursor
- P – put (paste) before cursor
- dd – delete (cut) a line
- 2dd – delete (cut) 2 lines
- dw – delete (cut) the characters of the word from the cursor position to the start of the next word
- D – delete (cut) to the end of the line
- d$ – delete (cut) to the end of the line
- x – delete (cut) character

## Exiting

- :w – write (save) the file, but don’t exit
- :w !sudo tee % – write out the current file using sudo
- :wq or :x or ZZ – write (save) and quit
- :q – quit (fails if there are unsaved changes)
- :q! or ZQ – quit and throw away unsaved changes

## Search and Replace

- `/pattern` – search for pattern
- `?pattern` – search backward for pattern
- `\vpattern` – ‘very magic’ pattern: non-alphanumeric characters are interpreted as special regex symbols (no escaping needed)
- n – repeat search in same direction
- N – repeat search in opposite direction
- :%s/old/new/g – replace all old with new throughout file
- :%s/old/new/gc – replace all old with new throughout file with confirmations
- :noh – remove highlighting of search matches
- Search in Multiple Files
- :vimgrep /pattern/ {file} – search for pattern in multiple files
- e.g.
- :vimgrep /foo/ **/*
- :cn – jump to the next match
- :cp – jump to the previous match
- :copen – open a window containing the list of matches

## Working With Multiple Files

- :e file – edit a file in a new buffer
- :bnext or :bn – go to the next buffer
- :bprev or :bp – go to the previous buffer
- :bd – delete a buffer (close a file)
- :ls – list all open buffers
- :sp file – open a file in a new buffer and split window
- :vsp file – open a file in a new buffer and vertically split window
- Ctrl + ws – split window
- Ctrl + ww – switch windows
- Ctrl + wq – quit a window
- Ctrl + wv – split window vertically
- Ctrl + wh – move cursor to the left window (vertical split)
- Ctrl + wl – move cursor to the right window (vertical split)
- Ctrl + wj – move cursor to the window below (horizontal split)
- Ctrl + wk – move cursor to the window above (horizontal split)

## Tabs

- :tabnew or :tabnew file – open a file in a new tab
- Ctrl + wT – move the current split window into its own tab
- gt or :tabnext or :tabn – move to the next tab
- gT or :tabprev or :tabp – move to the previous tab
- `#gt` – move to tab number #
- :tabmove # – move current tab to the #th position (indexed from 0)
- :tabclose or :tabc – close the current tab and all its windows
- :tabonly or :tabo – close all tabs except for the current one
- :tabdo command – run the command on all tabs (e.g. :tabdo q – closes all opened tabs)
## Even More
- If you're looking for even more help with vim, then check out the Vim Masterclass course. <https://courses.linuxtrainingacademy.com/course/vim-masterclass/>

# vim udemy

- `i, :, esc` - change modes
- `:wq`  - write + quit

## 3a. motion (nav.txt)

- `j,k`   - move down / up
- `h,l`   - move left / right
- `ctrl+f, ctrl+b` -move  page forward / back
- `w,b` - move word forward / back (by space or punctuation)
- `W,B` - move word forward / back ignoring puctuation
- `z+enter` - scroll to cursor
- `0` - move to beginning of line
- `^, $` - move to first / last non-space char of line
- `gg , G` - move to beginning / end of file
- `:$` - move to end of file using command mode
- `32gg`- jump to line 32
- `:32` - jump to line 32 using command mode
- `Ctrl g` - display file info in gutter
- `g Ctrl g`- display position info in gutter
- `:set ruler` - display row,col
- `:set noruler` - stop displaying row,col
- `:set ruler!` - toggle

## 3b. deleting (cutting) text (deletingtext.txt)

- count-operation-motion pattern
- `dw, db` - delete word / delete previous word
- `dl, dh` - delete char / previous char
- `x` - delete current char (shortcut for `dl`)
- `X` - delete previous char (shortcut for `dh`)
- `dh, dj, dk, dl, dw, db` delete current line to end / beggining,  down / up line, previous / next word -> `d5j` - delete multiple
- `d^, d$` - delete to the beginning / end of the line
- `D` shortcut for `d$`
- `dd` delete current line -> `3dd` delete 3 lines
- `5h, 5j, 5k, 5l, 5w, 5b` - navigate multiple
- `.` - repeat previous op

## 4. help (4)

- `:help` - top pane shows help file, bottom pane shows original file
- `Ctrl o, Ctrl i` - go to previous / next topic
- `Ctrl ]` - go to cursor selected topic
- `Ctrl ww` - toggle focus between top and bottom pane
- `:h :q ctrl d` - autocompletion suggestions for find help on a topic pattern. can tab to cycle through
- `:set wildmenu`

## 5. cut, copy, paste

- `dd` - delete (cut) line
- `p` - put (paste) after cursor
- `P` - put before cursor
- `y` - yank (copy) char
- `yy` - yank line
- `u` - undo
- `Ctrl r` - redo
- `:reg` - see all registers "0 .. "9
- `reg a b c` - see filtered list of registers
- `c` - change
- `s` - substitute ?
- `""` Any text that you delete (with d, c, s or x) or yank (with y) will be placed there and they will be shifter back
  - `""` - unnamed register
  - `"0` - most recent yank
  - `"1` - most recent delete / change
  - `"_` - delete text without storing in a register
- `"{char}y, "{char}p` - yank to register "x, paste from register "x
- `Ctrl-r r`- access register "r

- ## 4 read only registers

  - `".` - The last inserted text is stored
  - `"%` - the current file path
  - `":` -  the most recently executed command
  - `"#` -  the name of the alternate file

## 6A. transforming and substituting text

- `I` - enter edit mode at 1st non-space char of line
- `a` - append mode
- `A` - enter append mode at last non-space char of line
- `o` - insert mode on beggining of next line
- `O` - insert mode on beggining of previous line
- `R` - replace mode
- `r` - replace 1 char
- `cw` - change word - deletes word and changes to insert mode
- `c$` - change through to the end of the line
- `C` - shortcut for `c$`
- `"xcw` - change word, capture previous in register "x
- `cc` - change entire line
- `~` - toggle case of char
- `g~~` - toggle case of entire line
- `gUw` - change word to upper case
- `guw` - change word to lower case
- `gUU` - change line to upper case
- `guu` - change line to lower case
- `Shift j` - join line (append space to end of line + delete CR)
- `3 Shift j` - join 3 lines

## 6B. search, find and replace

- `f{char}` - find  next occurance of char on line (case sensitive)
- `F{char}` - find next occurance of char in reverse direction
- `;` - go to next occurance
- `,` - go to previous occurance
- `t{char}` , `T{char}` - like f, but positions the cursor before / after the char
- `dt{char}` - delete everything up to that char
- `/{chars}` - forward search - find occurances in doc. press `n` to jump to next, `N` to jump to previous
- `:set incsearch?` - see if setting for search highlight is on
- `:nohls` - disable highlight until next search
- find all `/and`, then `cw` to "&", then `n` to jump to next, then `.` to repeat change word
- `?{chars}` - reverse search
- `*` - search for word under the cursor
- `#` - reverse search for word under the cursor
- `d/{word}` - delete everything from cursor current position up until (but not including) search match
- `"ay/z` - yank all text up to char z into a" register
- `:{range}s/{old}/{new}/{flags}`
  - range can be a line number `22`, `1,5` first 5 lines of file, `$` last line, `.` current line, `.,$` current line to last line, `%` all lines
  - `:s/{old}/{new}/` - replace 1st occurance
  - `:s/{old}/{new}/g` - replace all occurances on the line
  - there is also pattern: `:/{pattern1}/,/{pattern2}/s/{old}/{new}/{flags}`
  - for visual mode, selection range is `'<,'>` eg `:'<,'>s/{old}/{new}/{flags}`
- `:set nu` - turn on line numbers. prefix with `no` to turn off, postfix with `!` to toggle

## 7A Text objects

- `daw` - delete a word (+ trailing dellimeter)
- `diw` - delete in word (trailing dellimeter is not affected)
- `das` - delete a sentence
- `dap` - delete a paragraph
- `ci[` - change everything within brackets of that type (leaves brackets intact). bracket types: `(, {, [, <` - could also be `cib` change in block
- `cit` - change in tag - useful for html editing !
- `ci"` change in quote. quote types: single, double, backticks

## 7B Macros

- `qa` - record macro into registry "a. `q` to stop recording. `:reg a` to see registry contents. `@a` to play back macro
- normalize your cursor in the macro (start with `0`, end with `j`) - then can apply to next line
- `5@b` - apply the macro in register "b 5 times
- uppercase letters ignore punctuation - eg record this macro: `02dWfL.` - move to first char, delete 2 words (ignoring punctuation), find L on line, repeat delete 2 words
- to append to a register (eg to "c), use uppercase capital `qC` - eg add `j`
- `Ctrl G` - get line count
- `:27,35 normal @b` - apply macro on specified line range. could use the range `.,$` for current line to end of file
- `"ap` - paste the contents of a macro (for editing). when ready, yank line back into a register `"ay$`
- `.viminfo` file - store history and non-empty registers
- `.vimrc` file - store initialization commands

## 8 Visual mode (selection)

- 3 versions !
  - `v` char visual mode
  - `V` line visual mode
  - `Ctrl-v` block visual mode (vim vertical selection)
- `o` - toggles cursor to opposite end of highlight
- `viw, vaw` - select word, `viq, vi(` select in quote block, bracket block
- `ciw, caw, ciq, ci(` - change
- can expand selection with `/f` search, `h,j,k,l`
- `U` - switch text to upercase, `u` (in visual mode) switch to lowercase
- `gv`- re-enter visual mode with same selection
- `Shift $` - select all the way to the end of all lines selected in block visual mode then `A` to append
- lowercase `a` or `i` dont work in visual mode - use `A, I`
- `>`, `<` - shift, unshift (in visual mode)
- `:set shiftwidth?` - shows shift size config
- `:set tabstop?` - shows tab size config
- `:set expandtab` - insert spaces instead of tab chars
- `:set list` - see tab chars as `^I`, externous whitespace as `$`
- in visual mode, selection range `'<,'>` is automatically prep-pended to `:` command mode prompt
- `:'<,'>center` - center the selection. shorthand: `:'<,'>ce` - other alignments: `le, ri`. can also provide column position: `:'<,'>le5`

## 9 vimrc

- rc at end of file stands for run commands
- `~/.vimrc`
- `set ruler` in .vimrc == `:set ruler` - each line is executed as a command
- `:version`
- `:checkhealth`
- `:set` - see all options that are not default
- `:h {option}` - see type, value for option
- `:h '{option}'` - see help for option
- `:set {option}?` - set if enabled
- `:set {option}!` - toggle
- `:set no{option}` - disable
- `:set {option}&` - reset to default
- `:e ~/.vimrc` - edit a file
- `:h option-list` - see all options - can also use `:options`

```
set history=1000

" show cursor position
set ruler

" see incomplete commands
set showcmd

" tab completion for commands
set wildmenu

" set min number of lines to keep above / below cursor
set scrolloff=5

" search options - highlight search matches, incrementally highlight as typing search pattern, overwrite case insensitive if search pattern contains uppercase
set hlsearch
set incsearch
set ignorecase
set smartcase

" show line numbers
set number

" make backup copy of file before saving
set backup

" dont linewrap words
set lbr

" autoindent (copy indentation from current line) - good for coding
set ai
" smart indenting - place close braces in correct place
set si

" set background tolight or dark
set bg=dark

" set color scheme
color=slate

" map key binding - similar to a macro
map <F3> i<ul><CR><Space><Space><li>blah</li><CR><ESC>0i</ul><Esc>kcit
```

- `:source ~/.vimrc`
- custom color schemes are stored in `~/.vim/colors/` - `it` - text object for inner tag
- `:mkvimrc` - generate config file from current session
- `vmap, nmap` - key mappings explicitly for visual mode, normal mode

## 10 vim buffers, windows

- `vim buf-ant.txt buf-bed.txt` - open multiple files at once
- `:e buf-cat.txt` - edit - open another file
- `:buffers` - see open files. also `:ls`. `:b3` will switch to buffer 3. - supports tab cycling
- `:b Ctrl-d` - shows list
- `:bn` cycle to next (short for `:bnext`), `:bp` cycle to previous, `:bf` first, `:bl` last
- buffer symbols: `%` current, `a` active, `h` hidden (loaded but not currently displayed), `#` previous buffer, `+` modified but not saved. can switch back using `:b#`
- `:qall`, close all `:wall` write all
-`:badd` open a file without switching to it
- `:bd3` - delete buffer 3 (close file - unloads buffer from memory). can delete a range of buffers `:1,3bd`, delete all `:%bd`
- `:bufdo set nu` turn on line numbers for all buffers
- `:E` open file explorer ! use `j,k` motions to select file and press enter to open

## 10B working with multiple windows

- `Ctrl w w` cycle through windows
- `:split`, `:vsplit` - can optionally specify filename
- `Ctrl wv, Ctrl ws, Ctrl wq, Ctrl wo` - split vertical, split horizontal, close current window, close all other windows
- `Ctrl w hjkl` - can use motion keys to navigate windows !
- `Ctrl w +, Ctrl w -` increase / decrease window height
- `Ctrl w +, Ctrl w <` increase / decrease window width
- `Ctrl w =` - set all windows to same size
- `Ctrl w _ , Ctrl w |` maximize height / width of window
- `Ctrl w r` - rotate windows
- `:ball` - open all buffers in windows
- `:windo set nu` - run cmd on all windows

## plugins

- `:set packpath`
- `:packadd` - load an optional package

## vim cheatsheet

<https://www.fprintf.net/vimCheatSheet.html>

- `f` - find next char. `3f{char}` move cursor to third next occurance of that char
- `A` - append at end of line
- `I` - insert at beginning of line
-
- `Yp` yank current line and put it on next line (copy it)
- `c()<Esc>P` - in visual mode, surround selection with brackets

## vim terminal

- `:term`
- `i` - enter edit mode
- `Ctrl \ Ctrl n` - escape edit mode