@echo off
title KOS-AGI Organism v9.0 — 60Hz Living Brain
color 0A

echo.
echo  ============================================================
echo    KOS-AGI ORGANISM v9.0 — 60Hz Thermodynamic Brain
echo  ============================================================
echo.
echo  [1] Start Organism Server (Dashboard + API)
echo  [2] Run KASM Universal AGI Tests (Movement / Recolor / Masking)
echo  [3] Run ARC Benchmark
echo  [4] Exit
echo.

set /p choice="  Select [1-4]: "

if "%choice%"=="1" goto server
if "%choice%"=="2" goto tests
if "%choice%"=="3" goto benchmark
if "%choice%"=="4" exit

:server
echo.
echo  [BOOT] Compiling Rust Kernel...
cd /d "%~dp0"
cd kos_rust && maturin develop --release 2>nul
cd /d "%~dp0"
echo  [BOOT] Starting 60Hz Organism...
echo  [BOOT] Dashboard: http://localhost:8090
echo.
python -u organism_api.py
pause
goto :eof

:tests
echo.
echo  ============================================================
echo    KASM VECTOR ALGEBRA TESTS
echo  ============================================================
echo.
cd /d "%~dp0"
echo  --- Universal AGI Tests (Movement + Recolor + 2D Grid) ---
echo.
python -u test_universal_agi.py
echo.
echo  --- Compositional Masking Tests (Selective Object Movement) ---
echo.
python -u test_masking.py
echo.
pause
goto :eof

:benchmark
echo.
echo  [BENCHMARK] Starting ARC-AGI evaluation...
cd /d "%~dp0"
python -u -c "import json, sys; sys.path.insert(0,'.'); from kos.brain import KOSBrain; b=KOSBrain(cache_dir='.cache/organism'); print(f'Loaded {len(b.kernel_nodes)} kernel nodes'); print('POST to http://localhost:8090/benchmark to run via API')"
echo.
echo  Start the server first (option 1), then POST to /benchmark
pause
goto :eof
