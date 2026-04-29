#!/bin/bash
# Quick status check for massless sweep
# Usage: bash scripts/check_status.sh

RESDIR=~/AdS4_Oscillons/results/massless

echo "=== Job status ==="
squeue -u $USER 2>/dev/null | head -20
echo

echo "=== CSV progress ==="
for f in $RESDIR/arclength_ell*_D3_*.csv; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    # Extract ell from filename
    ell=$(echo "$base" | sed 's/arclength_ell\([0-9]*\)_.*/\1/')
    lines=$(wc -l < "$f")
    data_lines=$((lines - 4))  # subtract header
    if [ $data_lines -lt 1 ]; then
        echo "  ell=$ell: no data yet"
        continue
    fi

    # Get last row
    last=$(tail -1 "$f")
    w=$(echo "$last" | awk -F, '{printf "%.4f", $2}')
    omega=$(echo "$last" | awk -F, '{printf "%.6f", $3}')
    M=$(echo "$last" | awk -F, '{printf "%.6e", $10}')
    res=$(echo "$last" | awk -F, '{printf "%.1e", $4}')

    # Check if fold detected (look for M decreasing in last 5 points)
    fold="no"
    if [ $data_lines -ge 5 ]; then
        # Get last 5 M values, check if any decrease
        M_vals=$(awk -F, 'NR>4 {print $10}' "$f" | tail -5)
        prev=""
        for val in $M_vals; do
            if [ -n "$prev" ]; then
                # Compare absolute values (M is negative)
                dec=$(awk "BEGIN {if ($val > $prev) print 1; else print 0}")
                if [ "$dec" = "1" ]; then
                    fold="YES"
                    break
                fi
            fi
            prev=$val
        done
    fi

    # Estimate state size and time
    case $ell in
        0)  n=961;  est_per_step="~15s" ;;
        2)  n=2881; est_per_step="~2min" ;;
        3)  n=3361; est_per_step="~3min" ;;
        4)  n=3841; est_per_step="~4min" ;;
        6)  n=4801; est_per_step="~7min" ;;
        8)  n=5761; est_per_step="~12min" ;;
        10) n=10081; est_per_step="~25min" ;;
        *)  n=0;    est_per_step="?" ;;
    esac

    # Rate estimate from data
    if [ $data_lines -ge 3 ]; then
        # Estimate steps to E_max: assume ~200 total needed
        remaining=$((200 - data_lines))
        [ $remaining -lt 0 ] && remaining=0
        echo "  ell=$ell: $data_lines pts, w=$w, omega=$omega, |M|=$M, res=$res, fold=$fold, ~${remaining} steps left ($est_per_step/step)"
    else
        echo "  ell=$ell: $data_lines pts (bootstrap), w=$w, omega=$omega ($est_per_step/step)"
    fi
done

echo
echo "=== Errors (non-empty .err files) ==="
found_err=0
for f in ~/AdS4_Oscillons/massless_*.err; do
    [ -f "$f" ] || continue
    if [ -s "$f" ]; then
        # Skip old cancelled job messages
        if grep -q "CANCELLED" "$f" 2>/dev/null; then
            continue
        fi
        echo "  $(basename $f): $(head -1 $f)"
        found_err=1
    fi
done
[ $found_err -eq 0 ] && echo "  (none)"

echo
echo "=== Estimated completion ==="
echo "  ell=0:  ~1 hour total (may already be at fold)"
echo "  ell=2:  ~3-6 hours with MKL"
echo "  ell=3:  ~5-8 hours"
echo "  ell=4:  ~8-14 hours"
echo "  ell=6:  ~1-2 days"
echo "  ell=8:  ~2-3 days"
echo "  ell=10: ~3-5 days"
