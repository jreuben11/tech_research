# gdb

`start`
`list`
`x/20i $pc` - examine 20 instructions starting at program counter
`step` - single step into
`next` - step over
`continue` - run until breakpoint / watchpoint
`print $1` - print variable value
`x $1` - examine variable address and raw byte values (as little endian hex)
`x/b $1` - examine variable byte by byte
`x/10b $1` - examine 10 bytes of variable
`print &xxx` - print pointer address of variable
`watch i` - hardware watchpoint - break on changes
`rwatch i` - hardware watchpoint - break on reads
`watch foo if foo > 0`
`watch foo thread 3`
`break 13` set breakpoint at this line
`info break` - see all breakpoints
`delete 7` - delete breakpoint 7
`backtrace` - see callstack
`frame 1` switch context to 1st frame
`set var xxx=23` - modify local
`ctrl-x-a` - switch to TUI mode
`ctrl-l` - repaint screen if stuck
`ctrl-x-2` - split view with dissasembly, registers
`ctrl-p` / `ctrl-n` - previous / next command
`python` - start interpreter inside gdb session !
`python gdb.execute('next')` - use python gdb module to execute gdb commands
`python gdb.parse_and_eval()`
`set print pretty on`
`call f(&x)` - run a function
import gdb.printing
`print $pc`
`record`
`break _exit`
`reverse-stepi`
`disass` disassembly
`print $sp` print the stack pointer
.gdbinit

```bash
set history save on
set print pretty on
set pagination off
set confirm off
```