# EEG & fNIRS Based Alzheimer's Disease Detection

**Team members:** Gangfeng Hu, Niko Hams, Sangdae Nam, Uli Prantz, Eric Ji, Matthew Zhou







## Introduction

### Signals: EEG



### Signals: fNIRS







### Tasks

**Resting State Task**:

+ Participants focused on a stationary white cross at the center of a black screen for 60 seconds.
+ The entire signal was treated as a single segment due to the absence of specific event triggers.

**Oddball Task**:

+ Participants viewed alternating circles (0.5 seconds) and an empty screen (1–1.5 seconds) on a black background.
+ Two types of circles were presented:
  + Yellow circles (target): Participants pressed the enter button.
  + Blue circles (non-target): No response was required.
+ This task involved detecting 28 events, with each event generating a 5-second segment for feature extraction.

**1-back Task**:

+ A random number from 1 to 3 was displayed for 1 second, followed by an empty screen for 1–1.5 seconds.
+ Participants pressed the enter button if the same number appeared consecutively.
+ Similar to the oddball task, this involved detecting 28 events, with 5-second segments extracted for each.

**Verbal Fluency Task**:

+ Included three phonemic and three semantic sub-tasks.
  + **Phonemic**: Participants generated words starting with a given letter.
  + **Semantic**: Participants produced words related to a provided category.
+ Each sub-task lasted 30 seconds, resulting in six segments (one for each sub-task) for feature extraction.



