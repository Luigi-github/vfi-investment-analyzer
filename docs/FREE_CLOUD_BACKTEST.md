# Free cloud execution for VFI Investment Analyzer (private repository)

## Recommended route: Oracle Cloud Always Free VM

This plan keeps the GitHub repository private and does not use GitHub Actions to execute Python. A cloud VM can run when your own computer is off. Oracle Always Free eligibility, available capacity, and terms can change; select only resources explicitly marked Always Free and verify the projected cost is $0 before creating them.

### 1. Create a free VM

1. Create or sign in to Oracle Cloud and open Compute → Instances.
2. Choose an Always Free eligible shape offered in your region and Ubuntu 22.04/24.04.
3. Verify the estimated monthly cost is $0. Do not select paid shapes, paid boot volumes, or other paid resources.
4. Connect to the VM over SSH.

Oracle may request card verification. If you do not want to provide a card, do not proceed with this provider; free notebooks exist but cannot guarantee unattended long-running execution.

### 2. Give the VM read-only access to the private repo

On the VM, generate a dedicated SSH key:

    ssh-keygen -t ed25519 -C "vfi-free-cloud-runner" -f ~/.ssh/vfi_repo_key -N ""
    cat ~/.ssh/vfi_repo_key.pub

In the private GitHub repository, open Settings → Deploy keys → Add deploy key. Paste the public key and leave “Allow write access” unchecked. Then configure SSH:

    cat >> ~/.ssh/config <<'EOF'
    Host github-vfi
      HostName github.com
      User git
      IdentityFile ~/.ssh/vfi_repo_key
      IdentitiesOnly yes
    EOF
    chmod 600 ~/.ssh/config
    ssh-keyscan github.com >> ~/.ssh/known_hosts
    git clone git@github-vfi:Luigi-github/vfi-investment-analyzer.git
    cd vfi-investment-analyzer

Never share or commit the private key.

### 3. Install Python dependencies

    sudo apt-get update
    sudo apt-get install -y git python3 python3-venv python3-pip
    cd ~/vfi-investment-analyzer
    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

### 4. Start the resumable 200-symbol run

The runner selects the intersection of Binance USDT spot and USD-M perpetual markets, ranked by combined 24-hour quote-volume ranks. It saves per-symbol outputs and atomically updates reports/free_cloud_backtest/checkpoint.json. If the VM reboots or a network/API error interrupts the run, rerun the same command; completed symbols are skipped.

    cd ~/vfi-investment-analyzer
    . .venv/bin/activate
    mkdir -p logs
    nohup .venv/bin/python -u scripts/run_resumable_crypto_backtests.py --limit 200 --min-volume 0 > logs/resumable-backtest.log 2>&1 &
    echo $! > logs/resumable-backtest.pid

Monitor progress:

    tail -f logs/resumable-backtest.log
    cat reports/free_cloud_backtest/checkpoint.json

For a small smoke test, use a separate output directory so the two-symbol universe is not reused for the full run:

    .venv/bin/python -u scripts/run_resumable_crypto_backtests.py --limit 200 --min-volume 0 --max-symbols 2 --output-dir reports/free_cloud_backtest_smoke

For the full run, omit --max-symbols and use the default reports/free_cloud_backtest output directory. The selected universe is saved on first launch and remains fixed for resume.

### 5. Automatically resume after a VM reboot

Create a systemd service. This assumes the VM's Linux username is ubuntu and the repository is in /home/ubuntu/vfi-investment-analyzer; adjust the paths and user if different.

    sudo tee /etc/systemd/system/vfi-backtest.service >/dev/null <<'EOF'
    [Unit]
    Description=VFI resumable Binance crypto backtest
    After=network-online.target
    Wants=network-online.target

    [Service]
    Type=simple
    User=ubuntu
    WorkingDirectory=/home/ubuntu/vfi-investment-analyzer
    ExecStart=/home/ubuntu/vfi-investment-analyzer/.venv/bin/python -u scripts/run_resumable_crypto_backtests.py --limit 200 --min-volume 0
    Restart=on-failure
    RestartSec=120
    Environment=PYTHONUNBUFFERED=1

    [Install]
    WantedBy=multi-user.target
    EOF
    sudo systemctl daemon-reload
    sudo systemctl enable --now vfi-backtest

Inspect status and logs:

    systemctl status vfi-backtest
    journalctl -u vfi-backtest -f

The service exits when all symbols finish, and restarts after a failure. It resumes from the saved checkpoint. If one symbol repeatedly fails, inspect its log and fix the cause before restarting.

### 6. Results

Main reports live in reports/free_cloud_backtest/:

- universe.txt: exact selected universe.
- checkpoint.json: progress and status.
- crypto_mtf_ranking.csv: multi-timeframe ranking rows.
- strategy_pnl_comparison.csv: strategy summaries.
- walk_forward_oos.csv and oos_trades.csv: out-of-sample results and trades.
- per_symbol/<SYMBOL>/: per-asset logs and outputs.

Copy reports to your computer using scp, or another secure file-transfer method. Avoid committing large reports or credentials into Git.

## Important limits

- This is a zero-cost design only while all selected resources remain eligible for Always Free. Verify provider terms and projected cost before provisioning.
- The runner requires internet access to Binance public market-data endpoints. Rate limits and outages can interrupt a run; checkpoints preserve completed symbols.
- Backtest results do not guarantee future performance. Review fees, slippage, funding, overlapping positions, and portfolio-level drawdown before considering live trading.
- GitHub Actions remains unused by this plan, so its billing block does not prevent Python execution on the VM.
