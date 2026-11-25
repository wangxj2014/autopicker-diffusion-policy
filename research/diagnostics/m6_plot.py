"""Plot measured M5/M6 distance and the actual hybrid event timeline."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research'))
from autopicker_dp.episode import Episode


def main():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [1.5, 1]}, sharex=True)
    for i, (left, right) in enumerate(axes):
        hybrid = Episode.load(ROOT / 'runs/autopicker/m6_hybrid_train_v1' / f'episode_{i:04d}')
        baseline = Episode.load(ROOT / 'runs/autopicker/m5_sample_long_train_v1' / f'episode_{i:04d}')
        for episode, label, color in ((baseline, 'Pure diffusion (M5)', '#8a94a6'),
                                      (hybrid, 'Event-gated hybrid (M6)', '#1768ac')):
            a = episode.arrays()
            captures = [e for e in episode.events if e['kind'] == 'capture']
            cutoff = captures[0]['time_s'] if captures else np.inf
            mask = a['time_s'] <= cutoff
            distance = np.linalg.norm(a['tip_world_m'] - a['goal_world_m'], axis=1) * 1000
            left.plot(a['time_s'][mask][::4], distance[mask][::4], color=color, lw=1.3, label=label)
        left.axhline(30, color='#b88400', ls='--', lw=.9, label='Precision entry (30 mm)')
        left.axhline(3, color='#27804f', ls=':', lw=1, label='Capture position limit (3 mm)')
        left.set_ylim(0, 170)
        left.set_ylabel('Distance to apple (mm)')
        left.grid(alpha=.15)
        target = hybrid.metadata['target_path'].split('/')[-1]
        left.set_title(f'{target} | ' + ('collected' if hybrid.outcome['success'] else 'captured; return timed out'), loc='left')
        a = hybrid.arrays()
        stages = [e for e in hybrid.events if e['kind'] == 'controller_stage']
        stage_values = np.zeros(len(a['time_s']))
        for value, stage in enumerate(stages, 1):
            stage_values[a['time_s'] >= stage['time_s']] = value
        right.step(a['time_s'], stage_values, where='post', color='#1768ac', label='Controller stage')
        for event in hybrid.events:
            if event['kind'] not in ('capture', 'collect'):
                continue
            color = '#27804f' if event['kind'] == 'capture' else '#9a4eb6'
            right.axvline(event['time_s'], color=color, ls='--', lw=1)
            right.annotate(f"{event['kind']}\n{event['time_s']:.2f} s", (event['time_s'], 2),
                           xytext=(-3 if event['time_s'] > 30 else 2, 9), textcoords='offset points',
                           ha='right' if event['time_s'] > 30 else 'left', color=color, fontsize=9)
        right.set_yticks([0, 1, 2], ['Diffusion', 'Precision IK', 'Return'])
        right.set_ylim(-.25, 3)
        right.grid(axis='x', alpha=.15)
        right.set_xlim(0, 35.5)
        if not hybrid.outcome['success']:
            right.axvline(35, color='#bc3434', lw=1)
            right.text(34.7, .4, '35 s timeout', color='#bc3434', ha='right')
        entry = stages[0]['time_s']
        capture = next(e['time_s'] for e in hybrid.events if e['kind'] == 'capture')
        left.axvspan(entry, capture, color='#b88400', alpha=.2)
    axes[0, 0].legend(loc='upper right', fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Simulation time (s)')
    fig.suptitle('Capture-gated hybrid: 3/3 seen targets captured, 2/3 collected within 35 s', fontsize=14)
    fig.text(.5, .012, 'Same frozen checkpoint. Hybrid uses oracle IK and classical return. '
             'Later episodes have different request seeds after early handoff.\n'
             'Before-capture distance is clipped at 170 mm for readability; '
             'orientation and attachment criteria are audited separately.', ha='center', fontsize=9, color='#505967')
    fig.tight_layout(rect=[0, .055, 1, .95], h_pad=2.0, w_pad=2.5)
    path = ROOT / 'docs/research/2026-11-23-m6-trajectory.png'
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    main()
