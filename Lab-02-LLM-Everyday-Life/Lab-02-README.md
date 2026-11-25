# Lab 02 — AI in Everyday Challenges: The Busy Student

**Module:** PUSL3205 — Large Language Models and Generative AI  
**Laboratory session:** 02 — AI in Everyday Challenges  
**Assigned group:** Group 5  
**Scenario:** The Busy Student  
**Activity:** Group-based LLM-assisted problem solving and early-stage product design  
**Submission type:** Individual learning and reflection README



## 1. Introduction and laboratory objective

Large Language Models can support practical problem solving by helping people explore ideas, clarify requirements, compare alternatives and communicate proposals. In this laboratory, the challenge was to use an LLM in an everyday situation rather than train a new language model. Our assigned scenario, **Group 5 — The Busy Student**, concerned students trying to manage academic work alongside their personal responsibilities.

The objectives were to understand the problem from the user's point of view; explore how prompting could support product ideation; consider a **product concept, name, key features, target users and visual mock-up**; and reflect on the decisions and limitations involved. The lab also included a group-presentation component explaining the problem, proposed product and role of the LLM.

My main learning goal was to understand how to use an LLM as a **thinking partner** while retaining human judgement over which ideas were feasible, useful and appropriate.

## 2. Problem analysis — what makes a student “busy”?

A university student's workload involves more than remembering deadlines. Lectures may occupy fixed hours, assignments often have different priorities, group meetings depend on other people, and personal responsibilities cannot simply be removed from the schedule. A useful solution must help students **decide what to do next** and recognise whether their plan is realistic.

| User challenge | Likely consequence | Design requirement |
|---|---|---|
| Deadlines scattered across different platforms | Important tasks may be overlooked | One consolidated task and deadline view |
| Several assignments due close together | Last-minute work and poor prioritisation | Clear urgency and importance indicators |
| Uncertain study duration | Overcrowded or unrealistic plans | Allow time estimates and free-time buffers |
| Group work and changing commitments | Planned sessions become outdated | Easy rescheduling and plan revision |
| Overlapping study and personal activities | Stress and reduced flexibility | Protect breaks and personal commitments |
| Too many alerts | Reminder fatigue | User-controlled, limited notifications |

These are **design assumptions based on the assigned scenario**, not findings from a user survey. In a real product-development project, interviews and usability testing would be needed to check which problems students experience most often.

### A representative user

Consider a student who attends lectures during the day, has a programming assignment due on Friday, needs to prepare for a group presentation and wants time for travel and rest. A calendar can display the events, but it does not automatically explain which remaining task deserves attention first. The product therefore needs to combine a timetable with an understandable prioritisation method.

## 3. Proposed product concept — StudyFlow

**Working name:** *StudyFlow — An LLM-Assisted Student Planner*.

*StudyFlow* is a **proposed concept used in this portfolio write-up** to make the design more concrete; it is not a claim that this name was formally chosen by the group during the original session.

**Concept:** A simple student-planning application where users can enter lectures, assignments, deadlines, group meetings and personal commitments. The system brings these into one view and can use an LLM to suggest a realistic study plan, explain priorities and help revise the plan when circumstances change.

**Target users:** Primarily university students managing multiple courses and commitments, especially students working on overlapping individual assignments and group projects. A later version might also support student mentors or academic advisors, but the initial design should stay focused on the student.

**Main value:** Reduce the effort involved in deciding what to work on next, without treating AI-generated schedules as compulsory or automatically modifying a student's calendar.

## 4. Features and priorities

| Feature | User benefit | Role of the LLM | Priority |
|---|---|---|---|
| Unified timetable and deadline list | Provides one overview of commitments | No LLM required for basic storage/display | Essential |
| Task priority suggestions | Helps decide what should be done first | Explain possible priorities using deadline, importance and estimated effort | Essential |
| Suggested study blocks | Turns free time into an actionable plan | Propose sessions in plain language within user-supplied constraints | Essential |
| Plan adjustment | Supports changes when a meeting or deadline moves | Suggest a revised schedule and describe trade-offs | Essential |
| Reminder preferences | Reduces forgotten tasks without excessive alerts | Optionally suggest useful reminder times; actual reminders use standard scheduling logic | Useful |
| Progress reflection | Helps the student identify unfinished work | Summarise tasks completed, postponed and still outstanding | Useful |
| Calendar integration and automatic notifications | Avoids repeated manual entry | LLM may interpret requests, but external writes require confirmation | Future extension |

### What should *not* depend on the LLM?

Deadline storage, date comparisons, calendar collision checks and notification delivery should use **deterministic application logic**. They need predictable behaviour. The LLM is better suited to generating explanations, suggesting alternatives and helping the student express preferences naturally. This separation reduces the risk of relying on an incorrect AI response for an important deadline.

### Simple prioritisation logic

A sensible starting point is to consider **deadline proximity**, **academic importance**, **estimated time required**, **dependencies** and **available study time**. Instead of claiming a perfect ranking formula, the interface should show *why* a task was suggested. For example: “Start the programming assignment because it is due sooner and needs approximately three hours; the reading task has a later deadline.” The student can accept or override the recommendation.

## 5. Using an LLM as a product-development partner

The purpose of prompting here is not simply to obtain a long list of app features. A better process is to **define the problem → generate alternatives → critique the suggestions → refine the proposal**.

### 5.1 Understanding the user

**Illustrative prompt for design exploration (not a saved classroom transcript):**

> Act as a student-experience researcher. A university student has lectures, two assignment deadlines, a group project and personal commitments. Identify the main scheduling problems, the decisions the student struggles to make, and the information an application would need. Do not assume the student is free every evening.

This is more useful than asking only “Give me an app idea” because it names the target user, situation and constraints.

### 5.2 Generating and narrowing product ideas

**Illustrative prompt:**

> Suggest three simple product approaches to help the busy student. Compare their user benefit, complexity, privacy concerns and likelihood of being used daily. Recommend a minimal first version with no more than four main functions.

The purpose is to compare options, not automatically adopt the LLM's first recommendation. An idea that requires too many integrations or sends constant notifications may sound impressive but be unsuitable for an initial prototype.

### 5.3 Critically challenging the design

**Illustrative prompt:**

> Critique the proposed student planner. What could go wrong if it suggests unrealistic study durations, ignores travel time, misunderstands an assignment deadline or sends too many alerts? Suggest changes that preserve user control.

This type of follow-up helps identify **failure cases** that may be missed during initial brainstorming. It also encourages critical evaluation of AI output, rather than assuming that a confident answer is a reliable answer.

### 5.4 Improving prompt quality

| Less effective prompt | More focused prompt | Why the second is better |
|---|---|---|
| “Make a timetable.” | “Suggest study blocks around these fixed lectures, leaving time for meals and travel.” | Identifies real constraints |
| “Which task is important?” | “Compare these tasks by due date, effort and impact; explain your reasoning.” | Makes the decision process inspectable |
| “Give some features.” | “Suggest four essential functions for a first prototype and explain what to postpone.” | Encourages realistic scope |

These examples demonstrate a **repeatable prompting strategy**. They do not constitute a quantitative prompt experiment or evidence of specific LLM responses, because no original prompt-response log is included in this portfolio folder.

## 6. Illustrative user journey

The following scenario demonstrates the intended behaviour of the *proposed* application, not a deployed feature test.

**Situation:** A student enters an assignment due Friday, a group meeting on Wednesday evening, lectures on Monday and Tuesday, and a personal commitment on Thursday afternoon.

1. The student records fixed events and assignment deadlines.
2. The application identifies available time slots without overwriting fixed commitments.
3. The LLM suggests breaking the assignment into research, implementation, testing and final review.
4. The student sees a short explanation for the suggested order of tasks.
5. When the Wednesday meeting changes, the student requests a revised plan.
6. The application presents the revision for approval rather than silently modifying the timetable.

**Expected design benefit:** The student receives a manageable plan and understands why the tasks were prioritised. Whether the plan truly saves time would need to be established through testing with users.

## 7. Low-fidelity interface mock-up

This text-only sketch is a **conceptual wireframe created for the portfolio**, not a screenshot of a built app or an original group prototype.

```text
+------------------------------------------------+
| StudyFlow                         Profile  [ ]  |
+------------------------------------------------+
| TODAY — Tuesday                                |
| Next class: Data Programming        10:00 AM   |
|                                                |
| DEADLINES                                      |
| [!] Programming assignment      Friday         |
| [ ] Group presentation          Next week       |
|                                                |
| SUGGESTED NEXT STEPS                           |
| 1. Work on assignment — 60 min                 |
|    Reason: nearest deadline, high effort       |
| 2. Review presentation outline — 30 min        |
|                                                |
| [Accept Plan]   [Edit Plan]   [Explain Why]     |
+------------------------------------------------+
| Home       Calendar       Tasks       Settings   |
+------------------------------------------------+
```

The prototype prioritises visibility of the **next commitment**, **approaching deadlines** and **explainable next actions**. The *Accept*, *Edit* and *Explain Why* controls reflect a central design choice: the student should stay in charge of the final plan.

A future clickable mock-up could test whether users understand these sections and find it easy to change the proposed schedule. No such usability test is claimed here.

## 8. Group activity and my learning process

This laboratory was organised as a **group product-development exercise**, so discussing different viewpoints was part of its educational purpose. The Busy Student scenario encouraged us to consider both the student's practical needs and how an LLM could contribute to the design process.

My strongest takeaway was that an LLM can help organise and compare ideas, but it cannot independently determine which features students need most. Human discussion is necessary to challenge assumptions, avoid unnecessary complexity and keep the proposed solution focused.

In future group activities, I would make the design process easier to assess by recording the initial problem statement, selected prompts and responses, alternatives considered, reasons for rejecting features and the final presentation materials. **Those historical group records are not included here**, so I have not invented group-member roles, meeting dates, classroom screenshots or feedback.

## 9. Critical observations, limitations and ethics

**Prompt quality affects usefulness.** The LLM needs sufficient context about the student's commitments and constraints. Vague requests are more likely to produce generic advice.

**Generated plans can be unrealistic.** An LLM might underestimate the time needed for a difficult assignment or forget travel and rest. The student must be able to edit every recommendation.

**Confident explanations are not always correct.** Recommendations should be checked against verified deadline and calendar data. The interface must not invent timetable entries or imply that university systems have been accessed when they have not.

**More features are not always better.** A first version should solve one clear problem before adding automatic integrations, complex analytics or frequent reminders.

**Privacy matters.** Study schedules can reveal personal routines and academic performance. A future system should minimise data collection, explain what is sent to an LLM provider, obtain consent for integrations and avoid sharing sensitive details unnecessarily.

**Automation needs approval.** Changing calendar events, sending messages or updating records should require explicit user confirmation. AI suggestions should support student agency rather than replace it.

**Evidence limitations.** This repository documents a design proposal and reflection. It does not demonstrate a completed application, measured productivity improvement, user interviews or a formal usability study.

## 10. How I would evaluate the idea

If *StudyFlow* were developed beyond the lab's ideation stage, I would evaluate it using a small, clearly documented student usability study rather than claiming success on the basis of an attractive prototype.

| Evaluation question | Proposed method | What success would mean |
|---|---|---|
| Can students identify their next important task? | Give participants a sample week and ask them to select the next action | Users can complete the task with little confusion |
| Are the suggested study blocks realistic? | Compare the suggestions with students' available hours and estimates | Plans respect fixed events and leave reasonable buffer time |
| Are priority explanations understandable? | Ask students to explain why a task was ranked first | Users can articulate the rationale and challenge it |
| Does the prototype feel overwhelming? | Short feedback survey and observation | Participants can find key functions without excessive steps |
| Are changes safely handled? | Test a moved meeting and a rejected plan update | No change is applied without confirmation |

These are **proposed evaluation activities**, not tests that have already been carried out.

## 11. Learning outcomes and personal reflection

Through this exercise, I developed a better understanding of the following skills:

- **Problem definition:** Identifying the real decision-making difficulties behind a broad scenario instead of treating every problem as a reminder problem.
- **Prompt design:** Providing an LLM with explicit context, constraints and output expectations, then asking critical follow-up questions.
- **Feature prioritisation:** Separating essential functions from attractive but unnecessary additions.
- **User-centred thinking:** Considering workload, flexibility, privacy and accessibility rather than only technical possibilities.
- **Critical evaluation:** Recognising that generated ideas need feasibility checks and eventual user feedback.
- **Communication and collaboration:** Presenting a concept through its problem, intended users, key features and simple interface.

The most important change in my understanding was recognising that a useful LLM application does not need AI for every function. Reliable calendar calculations and deadline checking should be conventional software features, while the LLM can help explain options and turn a busy list of tasks into a more understandable plan. This distinction makes a proposed product more realistic and safer.

## 12. Completion and evidence statement

| Lab activity | Evidence in this README |
|---|---|
| Identify the Busy Student problem | Problem analysis and representative user |
| Explain a product concept and name | Proposed *StudyFlow* concept |
| Identify target users | University-student target group |
| Consider key features | Prioritised feature table |
| Illustrate an interface | Text-based conceptual wireframe |
| Explain LLM involvement | Prompt examples and design process |
| Reflect on limitations and learning | Sections 8–11 |
| Original group presentation, historical prompt log and classroom mock-up | **Not attached; add authentic evidence if available** |
| Working application or formal user-testing results | **Not part of this documented work** |

## 13. Conclusion

Lab 02 demonstrated that an LLM can contribute to the early stages of product design by helping clarify a real-world problem, suggest alternative approaches and challenge an initial proposal. The Busy Student scenario showed that the most useful solution would not simply store deadlines: it would help students make informed choices about priorities and time while leaving them in control of their schedules.

Developing the *StudyFlow* concept for this portfolio strengthened my understanding of user needs, prompt clarity, feature prioritisation and responsible AI use. It also reinforced an important limitation: a well-explained concept is **not the same as a tested product**. To move beyond the laboratory's ideation task, the next steps would be to validate the assumptions with students, create a clickable mock-up, test realistic scenarios and revise the design using actual feedback.

## 14. References and AI-use acknowledgement

- PUSL3205 — *Large Language Models and Generative AI*, Lab 02: AI in Everyday Challenges (module-provided laboratory sheet).
- PUSL3205 — *Enhanced Assessment Brief 2026–27* (module-provided assessment guidance).
