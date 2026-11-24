# Instructor Delivery Script: Lecture 27 (Research Archaeology: Stages 01 to 09)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Epistemology of Scientific Auditing: Code vs. Documentation
00:20 - 00:45 | Autopsy of the Stage 01 Urdu Baseline Disaster (88.4% WER)
00:45 - 01:10 | Uncovering the 22.05 kHz Silent Resampling Bug
01:10 - 01:25 | The Architecture Pivot: Why CTC Won Over Whisper in Stage 04
01:25 - 01:30 | Synthesis & Lessons for Research Engineers
```

---

## Instructor Notes & Dialogue

### 00:00 - 00:20 | Code vs. Documentation
- Address the class:
  > "When you join an AI research lab, the former PhD student's README will say: 'Everything works perfectly! Model achieves 15% WER.'
  > You run the script: it crashes on line 4.
  > You check the git log: the commit message says 'wip temp hack'.
  > You check the checkpoint: the weights are NaNs.
  > In empirical research, **documentation is an aspiration; source code, execution logs, and checkpoint hashes are physical truth**.
  > Today, we conduct a forensic excavation of the first nine stages of this project.
  > We will study why our predecessors failed, how they debugged their failures, and how those hard lessons built the foundation for modern Kashmiri speech AI."
