import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT = Path("docs/PROJECT_REPORT.pdf")
SUMMARY = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))

styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="CoverTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=30,
        textColor=colors.HexColor("#102A43"),
        alignment=TA_CENTER,
        spaceAfter=14,
    )
)
styles.add(
    ParagraphStyle(
        name="CoverSub",
        parent=styles["Normal"],
        fontSize=12,
        leading=18,
        textColor=colors.HexColor("#486581"),
        alignment=TA_CENTER,
    )
)
styles["Heading1"].textColor = colors.HexColor("#102A43")
styles["Heading1"].spaceBefore = 8
styles["Heading1"].spaceAfter = 8
styles["Heading2"].textColor = colors.HexColor("#1F5A7A")
styles["BodyText"].leading = 14
styles["BodyText"].spaceAfter = 7


def page_frame(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E2EC"))
    canvas.line(18 * mm, 17 * mm, 192 * mm, 17 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#627D98"))
    canvas.drawString(18 * mm, 11 * mm, "Checkpoint-Aware GPU Job Scheduler")
    canvas.drawRightString(192 * mm, 11 * mm, f"Page {document.page}")
    canvas.restoreState()


def metric_table():
    data = [
        ["Metric", "FCFS", "Checkpoint"],
        ["Median average short-job wait", "8.82 s", "4.11 s"],
        ["Median training completion", "5.16 s", "14.27 s"],
        ["Median wait reduction", "-", "52.2%"],
        ["Median additional training time", "-", "9.10 s"],
    ]
    table = Table(data, colWidths=[85 * mm, 40 * mm, 40 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F5A7A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F0F4F8")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BCCCDC")),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


story = [
    Spacer(1, 34 * mm),
    Paragraph("Checkpoint-Aware<br/>GPU Job Scheduler", styles["CoverTitle"]),
    Paragraph(
        "A C++ and PyTorch prototype for serving latency-sensitive inference jobs during long training runs",
        styles["CoverSub"],
    ),
    Spacer(1, 24 * mm),
    metric_table(),
    Spacer(1, 18 * mm),
    Paragraph("Prepared by Karan Ragireddy", styles["CoverSub"]),
    Paragraph("Five-run controlled benchmark | November 2026", styles["CoverSub"]),
    PageBreak(),
    Paragraph("1. Problem and objective", styles["Heading1"]),
    Paragraph(
        "A shared GPU can be occupied by a long training job while a short inference request waits. The project evaluates whether cooperative checkpointing can release the device, serve queued work, and resume training without changing the final model state.",
        styles["BodyText"],
    ),
    Paragraph("2. System design", styles["Heading1"]),
    Paragraph(
        "The C++ scheduler launches and observes Linux processes using fork, exec, and waitpid. A PyTorch worker checks for a pause request between optimizer steps. It saves the completed step, model weights, and optimizer state to a temporary file, then atomically publishes the checkpoint and exits. The scheduler serves queued inference work and launches a fresh worker in resume mode.",
        styles["BodyText"],
    ),
    Paragraph("Execution sequence", styles["Heading2"]),
    Table(
        [
            ["1", "Train", "Long-running worker owns the device."],
            ["2", "Request", "Short job enters the queue."],
            ["3", "Checkpoint", "Training state is atomically committed."],
            ["4", "Serve", "Trainer exits and inference runs."],
            ["5", "Resume", "Training restarts from the committed step."],
        ],
        colWidths=[12 * mm, 32 * mm, 120 * mm],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (1, -1), colors.HexColor("#D9EAF2")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BCCCDC")),
                ("FONTNAME", (0, 0), (1, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        ),
    ),
    Spacer(1, 7 * mm),
    Paragraph("3. Correctness verification", styles["Heading1"]),
    Paragraph(
        "The correctness notebook compares a resumed training run with an uninterrupted reference and checks every final parameter using PyTorch tolerance. The queue benchmark also confirms that both short jobs execute under each policy. The submitted run produced the expected PASS result.",
        styles["BodyText"],
    ),
    PageBreak(),
    Paragraph("4. Benchmark results", styles["Heading1"]),
    Paragraph(
        "Each of five runs submitted two inference jobs during a paced 100-step training workload. The table reports medians across the five runs.",
        styles["BodyText"],
    ),
    metric_table(),
    Spacer(1, 8 * mm),
    Paragraph("Interpretation", styles["Heading2"]),
    Paragraph(
        "Checkpoint scheduling reduced the median average wait from 8.82 seconds to 4.11 seconds. Both queued jobs started sooner in every recorded run. The median 9.10-second increase in training completion includes checkpoint writing, worker termination, inference-process startup, inference execution, and training restart.",
        styles["BodyText"],
    ),
    Paragraph("5. Limitations", styles["Heading1"]),
    Paragraph(
        "The experiment uses a synthetic model, controlled sleep-based pacing, a filesystem queue, and cooperative workers. The timings describe this prototype only. The current logs do not separate checkpoint cost, GPU release, inference startup, inference execution, and restart cost. The benchmark policy order is fixed in the supplied notebook.",
        styles["BodyText"],
    ),
    Paragraph("6. Next engineering milestones", styles["Heading1"]),
    Paragraph(
        "Add phase-level timestamps, alternate policy order, evaluate real training and inference workloads, add priority aging, verify GPU-memory release, and replace queue files with a Unix-domain socket interface. Multi-GPU placement and crash recovery would complete the transition from a demonstrator to a reusable scheduler.",
        styles["BodyText"],
    ),
    Paragraph("7. Reproduction", styles["Heading1"]),
    Paragraph(
        "Upload the two notebooks from the repository to Google Colab. Run the correctness notebook first, then the queue benchmark. The repository also contains the C++ source, PyTorch worker, five raw run files, result summarizer, tests, and browser-based GitHub upload guide.",
        styles["BodyText"],
    ),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
document = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    rightMargin=18 * mm,
    leftMargin=18 * mm,
    topMargin=18 * mm,
    bottomMargin=23 * mm,
    title="Checkpoint-Aware GPU Job Scheduler",
    author="Karan Ragireddy",
)
document.build(story, onFirstPage=page_frame, onLaterPages=page_frame)
print(OUTPUT)
