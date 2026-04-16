@echo off
cd /d "%~dp0"
echo [overnight] Checking prerequisites...
if not exist "bot.exe"                 ( echo ERROR: bot.exe not found & pause & exit /b 1 )
if not exist "hdgl_lattice_corpus.bin" ( echo ERROR: hdgl_lattice_corpus.bin missing & pause & exit /b 1 )
if not exist "hdgl_corpus_seeder.exe"  ( echo ERROR: hdgl_corpus_seeder.exe missing & pause & exit /b 1 )
echo [overnight] All prerequisites OK.
echo.
echo [overnight] Launching brainwave monitor in new window...
start "Brainwave Monitor" cmd /k "cd /d %~dp0 && python -u ..\pipeline\brainwave_chladni.py --log ..\pipeline\sft\self_dialogue_log.jsonl --fps 4 --bot .\bot.exe --train ..\pipeline\sft\train.jsonl"
echo.
echo [overnight] Starting self-dialogue (unlimited epochs, 8h wall-clock)...
echo [overnight] Using sparse_eval (cold-zone). Reseed every 5 epochs. pvar-thresh 7.0 crescendo 0.3.
echo.
python -u "..\pipeline\self_dialogue.py" ^
  --epochs 0 ^
  --hours 8 ^
  --pvar-thresh 7.0 ^
  --crescendo 0.3 ^
  --min-new 0 ^
  --live ^
  --residual-sort ^
  --hdgl-load ".\hdgl_lattice_corpus.bin" ^
  --ngrams-load ".\hdgl_lattice_corpus_ngrams.bin" ^
  --reseed-every 5 ^
  --bot ".\bot.exe" ^
  --eval "..\pipeline\sft\sparse_eval.jsonl" ^
  --train "..\pipeline\sft\train.jsonl" ^
  --log "..\pipeline\sft\self_dialogue_log.jsonl"
echo.
echo [overnight] Self-dialogue complete. Exit code: %ERRORLEVEL%
pause
