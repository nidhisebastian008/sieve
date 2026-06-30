Output demo.gif

Set Shell "bash"
Set FontSize 14
Set Width 900
Set Height 550
Set Theme "Dracula"
Set Padding 20
Set TypingSpeed 40ms

Hide
Type "cd ~/sieve && source .venv/bin/activate"
Enter
Sleep 500ms
Show

Type "# sieve — production traces → training data pipeline"
Sleep 1s
Enter
Sleep 300ms

Type "sieve ingest /tmp/sample.jsonl"
Sleep 500ms
Enter
Sleep 1.5s

Type "sieve score"
Sleep 500ms
Enter
Sleep 1.5s

Type "sieve stats"
Sleep 500ms
Enter
Sleep 2s

Type "sieve dataset create v1.0 --min-quality 0.6"
Sleep 500ms
Enter
Sleep 1.5s

Type "sieve dataset list"
Sleep 500ms
Enter
Sleep 2s

Type "sieve dataset export v1.0 --output train.jsonl"
Sleep 500ms
Enter
Sleep 1.5s

Type "sieve train v1.0 --export-path train.jsonl"
Sleep 500ms
Enter
Sleep 2s

Sleep 2s
