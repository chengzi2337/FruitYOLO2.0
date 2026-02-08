import argparse
import subprocess
import time
import os
import signal
import sys
try:
    import pandas as pd
except Exception:
    pd = None


def read_metric(csv_path, metric, window=3):
    if not os.path.exists(csv_path):
        return None
    if pd:
        df = pd.read_csv(csv_path)
        # accept common column names
        col = None
        for c in df.columns:
            if metric in c or c.replace('_','') == metric.replace('_',''):
                col = c
                break
        if col is None:
            return None
        s = df[col].rolling(window, min_periods=1).mean()
        return s
    else:
        # fallback: simple CSV parse
        import csv
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return None
            if metric not in rows[0]:
                # try some variants
                for k in rows[0].keys():
                    if metric in k:
                        metric = k
                        break
            vals = []
            for r in rows:
                try:
                    vals.append(float(r.get(metric, 'nan')))
                except Exception:
                    vals.append(float('nan'))
            # simple moving average
            out = []
            for i in range(len(vals)):
                window_vals = [v for v in vals[max(0, i-window+1):i+1] if not (v!=v)]
                out.append(sum(window_vals)/len(window_vals) if window_vals else float('nan'))
            return out


def build_command(base_cmd, lr=None, resume=False):
    cmd = base_cmd
    # Ultralytics CLI expects lr0= or lrf=, not lr=
    if lr is not None:
        cmd += f" lr0={lr}"
    if resume:
        cmd += " resume=True"
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True, help='base shell command to run (powerShell), e.g. "conda activate pytorch; yolo train ..."')
    p.add_argument('--metric', default='fitness', help='metric to monitor (fitness or mAP_50 or mAP_50-95 etc)')
    p.add_argument('--min_delta', type=float, default=0.0005)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--min_epochs', type=int, default=10)
    p.add_argument('--smooth', type=int, default=3)
    p.add_argument('--check_interval', type=int, default=30, help='seconds between checks')
    p.add_argument('--project', default='runs/detect')
    p.add_argument('--name', default='resplit_train_plateau')
    p.add_argument('--lr_reduce_factor', type=float, default=0.2)
    p.add_argument('--max_reductions', type=int, default=1)
    p.add_argument('--initial_lr', type=float, default=None)
    args = p.parse_args()

    run_dir = os.path.join(args.project, args.name)
    metrics_csv = os.path.join(run_dir, 'metrics.csv')
    results_csv = os.path.join(run_dir, 'results.csv')

    lr = args.initial_lr
    reductions = 0

    # function to start training subprocess
    def start_train(cmd):
        # run in powershell so conda activation works
        if os.name == 'nt':
            popen = subprocess.Popen(["powershell", "-Command", cmd], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            popen = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        return popen

    current_cmd = build_command(args.base, lr=lr, resume=False)
    print('Start command:', current_cmd)
    proc = start_train(current_cmd)
    start_time = time.time()

    try:
        while True:
            time.sleep(args.check_interval)
            # if process ended, exit
            if proc.poll() is not None:
                print('Training process exited with', proc.returncode)
                break

            # prefer metrics.csv but fallback to results.csv
            csv_to_use = metrics_csv if os.path.exists(metrics_csv) else (results_csv if os.path.exists(results_csv) else None)
            if csv_to_use is None:
                print('No metrics file found yet; waiting...')
                time.sleep(args.check_interval)
                continue
            s = read_metric(csv_to_use, args.metric, window=args.smooth)
            if s is None:
                print('metrics.csv not ready yet; waiting...')
                continue

            if pd:
                sm = s.dropna()
                if sm.empty:
                    continue
                best = sm.max()
                # require at least min_epochs rows
                epochs_done = len(sm)
                recent = sm.tail(args.patience)
                print(f'epochs={epochs_done} best={best:.6f} recent_tail={list(recent.round(6))}')
                if epochs_done < args.min_epochs:
                    continue
                # check plateau
                if ((best - recent) <= args.min_delta).all():
                    print('Plateau detected')
                    if reductions < args.max_reductions:
                        # reduce LR and resume
                        reductions += 1
                        if lr is None:
                            # can't determine lr: set a default
                            lr = 0.01
                        lr = lr * args.lr_reduce_factor
                        print(f'Reducing LR -> {lr}, restarting with resume')
                        # send ctrl-break to allow YOLO to save
                        try:
                            if os.name == 'nt':
                                proc.send_signal(signal.CTRL_BREAK_EVENT)
                            else:
                                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        except Exception as e:
                            print('Failed to send interrupt to process:', e)
                            proc.terminate()
                        proc.wait()
                        # restart with resume and new lr
                        current_cmd = build_command(args.base, lr=lr, resume=True)
                        proc = start_train(current_cmd)
                        continue
                    else:
                        print('Max LR reductions reached — stopping training')
                        try:
                            if os.name == 'nt':
                                proc.send_signal(signal.CTRL_BREAK_EVENT)
                            else:
                                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        except Exception:
                            proc.terminate()
                        proc.wait()
                        break
            else:
                # fallback simple list
                sm = s
                epochs_done = len(sm)
                if epochs_done < args.min_epochs:
                    continue
                best = max(sm)
                recent = sm[-args.patience:]
                print(f'epochs={epochs_done} best={best:.6f} recent_tail={recent}')
                if all((best - v) <= args.min_delta for v in recent):
                    print('Plateau detected (fallback)')
                    if reductions < args.max_reductions:
                        reductions += 1
                        if lr is None:
                            lr = 0.01
                        lr = lr * args.lr_reduce_factor
                        print(f'Reducing LR -> {lr}, restarting with resume')
                        try:
                            if os.name == 'nt':
                                proc.send_signal(signal.CTRL_BREAK_EVENT)
                            else:
                                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        except Exception:
                            proc.terminate()
                        proc.wait()
                        current_cmd = build_command(args.base, lr=lr, resume=True)
                        proc = start_train(current_cmd)
                        continue
                    else:
                        print('Max LR reductions reached — stopping training')
                        try:
                            if os.name == 'nt':
                                proc.send_signal(signal.CTRL_BREAK_EVENT)
                            else:
                                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        except Exception:
                            proc.terminate()
                        proc.wait()
                        break
    except KeyboardInterrupt:
        print('Controller interrupted by user, terminating training')
        try:
            proc.terminate()
        except Exception:
            pass


if __name__ == '__main__':
    main()
