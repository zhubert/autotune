"""
Visual dashboard renderers using Rich

Beautiful terminal UI for displaying progress, models, and recommendations.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from datetime import datetime
from typing import List

from .progress_tracker import ProgressTracker
from .models import Model, PhaseStatus


console = Console()

# Phase names for display
PHASE_NAMES = {
    "phase_1_sft": "Phase 1: Supervised Fine-Tuning (SFT)",
    "phase_2_reward": "Phase 2: Reward Modeling",
    "phase_3_dpo": "Phase 3: Direct Preference Optimization (DPO)",
    "phase_4_rlhf": "Phase 4: RLHF with PPO"
}


def render_progress_dashboard(tracker: ProgressTracker):
    """Display comprehensive progress dashboard"""

    # Header
    console.print()
    header = Text()
    header.append("🎓 Your Training Journey", style="bold cyan")

    profile = tracker.get_user_profile()
    if profile.experience_level.value == "beginner":
        header.append(" 🌱", style="green")
    elif profile.experience_level.value == "intermediate":
        header.append(" 🌿", style="yellow")
    else:
        header.append(" 🌳", style="blue")

    console.print(Panel(header, border_style="cyan"))
    console.print()

    # Check for active training
    active = tracker.get_active_training()
    if active:
        render_active_training(active)
        console.print()

    # Learning phases (if enabled)
    if profile.learning_path_enabled:
        render_learning_phases(tracker)
        console.print()

    # Model registry summary
    render_model_summary(tracker.registry)
    console.print()

    # Recent activity
    render_recent_activity(tracker)
    console.print()

    # Recommendations
    recommendations = tracker.recommender.generate_recommendations()
    render_recommendations(recommendations)
    console.print()


def render_active_training(session):
    """Display active training progress"""
    progress_pct = session.progress_pct

    console.print("[bold yellow]⏳ Training In Progress[/bold yellow]")
    console.print()
    console.print(f"  Method: [cyan]{session.method.upper()}[/cyan]")
    if session.model_id:
        console.print(f"  Model: [cyan]{session.model_id}[/cyan]")
    console.print(f"  Progress: {progress_pct:.1f}% ({session.current_step:,}/{session.total_steps:,} steps)")

    # Progress bar
    if session.total_steps > 0:
        bar_width = 40
        filled = int(bar_width * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        console.print(f"  [{bar}] {progress_pct:.1f}%")


def render_learning_phases(tracker: ProgressTracker):
    """Display learning phase progress"""
    phases = tracker.state.get("learning_phases", {})

    console.print("[bold cyan]📚 Learning Path Progress[/bold cyan]")
    console.print()

    phase_order = ["phase_1_sft", "phase_2_reward", "phase_3_dpo", "phase_4_rlhf"]

    for phase_id in phase_order:
        if phase_id not in phases:
            continue

        phase_data = phases[phase_id]
        status = phase_data.get("status")

        # Status icon
        if status == PhaseStatus.COMPLETED.value:
            icon = "[green]✅[/green]"
        elif status == PhaseStatus.IN_PROGRESS.value:
            icon = "[yellow]🔄[/yellow]"
        elif status == PhaseStatus.AVAILABLE.value:
            icon = "[cyan]⏳[/cyan]"
        else:  # LOCKED
            icon = "[dim]🔒[/dim]"

        # Phase name
        name = PHASE_NAMES.get(phase_id, phase_id)

        # Details
        details = ""
        if status == PhaseStatus.COMPLETED.value:
            completed_at = phase_data.get("completed_at")
            if completed_at:
                age = _format_time_ago(completed_at)
                details = f"[dim]completed {age}[/dim]"
        elif status == PhaseStatus.LOCKED.value:
            requirements = phase_data.get("unlock_requirements", [])
            req_names = [PHASE_NAMES.get(r, r) for r in requirements]
            details = f"[dim]requires: {', '.join([r.split(':')[0] for r in req_names])}[/dim]"
        elif status == PhaseStatus.AVAILABLE.value:
            details = "[cyan]ready to start[/cyan]"

        console.print(f"  {icon} {name} {details}")


def render_model_summary(registry):
    """Display summary of models in registry"""
    stats = registry.get_stats()

    console.print("[bold cyan]🗂️  Model Registry[/bold cyan]")
    console.print()

    # Quick stats
    console.print(f"  Total Models: [cyan]{stats['total_models']}[/cyan]")

    if stats['training'] > 0:
        console.print(f"  Currently Training: [yellow]{stats['training']}[/yellow]")

    if stats['ready_for_dpo'] > 0:
        console.print(f"  Ready for DPO: [green]{stats['ready_for_dpo']}[/green]")

    if stats['ready_for_rlhf'] > 0:
        console.print(f"  Ready for RLHF: [green]{stats['ready_for_rlhf']}[/green]")

    if stats['reward_models'] > 0:
        console.print(f"  Reward Models: [magenta]{stats['reward_models']}[/magenta]")


def render_recent_activity(tracker: ProgressTracker):
    """Display recent training activity"""
    recent = tracker.get_recent_sessions(limit=5)

    if not recent:
        return

    console.print("[bold cyan]📅 Recent Activity[/bold cyan]")
    console.print()

    for session in recent:
        status_icon = {
            "completed": "[green]✓[/green]",
            "in_progress": "[yellow]⏳[/yellow]",
            "failed": "[red]✗[/red]",
            "abandoned": "[dim]○[/dim]"
        }.get(session.status.value, "○")

        method = session.method.upper()

        if session.status == "completed":
            duration = session.duration_minutes
            console.print(f"  {status_icon} {method} - completed in {duration:.0f} min")
            if session.checkpoint_path:
                console.print(f"     [dim]{session.checkpoint_path}[/dim]")
        elif session.status == "in_progress":
            console.print(f"  {status_icon} {method} - {session.progress_pct:.0f}% complete")
        else:
            console.print(f"  {status_icon} {method} - {session.status.value}")


def render_recommendations(recommendations):
    """Display personalized recommendations"""

    lines = []

    # Primary recommendation
    lines.append(f"[bold]1. {recommendations.suggested_action_title}[/bold]")
    lines.append(f"   {recommendations.reason}")
    lines.append("")

    # Alternatives
    if recommendations.alternatives:
        lines.append("[bold]Or try:[/bold]")
        for i, alt in enumerate(recommendations.alternatives, 2):
            lines.append(f"{i}. {alt.title}")
            lines.append(f"   [dim]{alt.description}[/dim]")

    content = "\n".join(lines)

    console.print(Panel(
        content,
        title="🎯 Recommended Next Steps",
        border_style="yellow"
    ))


def render_model_browser(registry):
    """Display interactive model browser"""

    console.print()
    console.print("[bold cyan]🗂️  Model Registry Browser[/bold cyan]")
    console.print()

    # Group models by base model
    by_base = {}
    for model in registry.get_all_models():
        base = model.base_model
        if base not in by_base:
            by_base[base] = []
        by_base[base].append(model)

    if not by_base:
        console.print("[dim]No models registered yet.[/dim]")
        console.print()
        return

    # Create tree view
    tree = Tree("📦 Models")

    for base_model, models in by_base.items():
        base_node = tree.add(f"[bold cyan]{base_model}[/bold cyan] ({len(models)} models)")

        for model in models:
            status_icon = {
                "ready": "✅",
                "training": "🔄",
                "failed": "❌"
            }.get(model.status.value, "○")

            # Build model display name
            tags_str = f"[dim]({', '.join(model.tags[:2])})[/dim]" if model.tags else ""
            model_node = base_node.add(
                f"{status_icon} [bold]{model.id}[/bold] {tags_str}"
            )

            # Show lineage/pipeline
            pipeline = " → ".join([entry.method.upper() for entry in model.lineage])
            model_node.add(f"Pipeline: {pipeline}")

            # Show capabilities
            if model.capabilities:
                caps = ", ".join(model.capabilities)
                model_node.add(f"Capabilities: [green]{caps}[/green]")

            # Show checkpoint
            latest_checkpoint = model.get_latest_checkpoint()
            if latest_checkpoint:
                model_node.add(f"[dim]{latest_checkpoint}[/dim]")

            # Show notes if available
            if model.notes:
                model_node.add(f"[italic]{model.notes}[/italic]")

    console.print(tree)
    console.print()

    # Show stats
    stats = registry.get_stats()
    render_registry_stats(stats)


def render_registry_stats(stats):
    """Display registry statistics table"""

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Stat", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Total Models", str(stats["total_models"]))
    table.add_row("Currently Training", str(stats["training"]))
    table.add_row("Ready for DPO", str(stats["ready_for_dpo"]))
    table.add_row("Ready for RLHF", str(stats["ready_for_rlhf"]))
    table.add_row("Reward Models", str(stats["reward_models"]))

    console.print(Panel(table, title="📊 Registry Stats", border_style="cyan"))
    console.print()


def render_model_details(model: Model):
    """Display detailed information about a specific model"""

    console.print()
    console.print(Panel(
        f"[bold cyan]{model.id}[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    # Basic info
    info_table = Table(show_header=False, box=None)
    info_table.add_column("Field", style="cyan", width=20)
    info_table.add_column("Value", style="white")

    info_table.add_row("Base Model", model.base_model)
    info_table.add_row("Status", model.status.value)
    info_table.add_row("Type", model.model_type)
    info_table.add_row("Created", _format_time_ago(model.created_at))

    if model.tags:
        info_table.add_row("Tags", ", ".join(model.tags))

    if model.capabilities:
        info_table.add_row("Capabilities", ", ".join(model.capabilities))

    if model.notes:
        info_table.add_row("Notes", model.notes)

    console.print(info_table)
    console.print()

    # Training lineage
    console.print("[bold]Training Lineage:[/bold]")
    console.print()

    for i, entry in enumerate(model.lineage, 1):
        status_icon = "✓" if entry.status == "completed" else "⏳"
        console.print(f"{i}. {status_icon} [cyan]{entry.method.upper()}[/cyan]")

        if entry.dataset:
            console.print(f"   Dataset: {entry.dataset}")

        if entry.completed_at:
            console.print(f"   Completed: {_format_time_ago(entry.completed_at)}")

        console.print(f"   Checkpoint: [dim]{entry.checkpoint}[/dim]")

        if entry.metrics:
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in entry.metrics.items()])
            console.print(f"   Metrics: {metrics_str}")

        console.print()


def _format_time_ago(iso_timestamp: str) -> str:
    """Format ISO timestamp as 'X time ago'"""
    try:
        created = datetime.fromisoformat(iso_timestamp)
        now = datetime.now()
        delta = now - created

        if delta.days > 0:
            if delta.days == 1:
                return "1 day ago"
            elif delta.days < 7:
                return f"{delta.days} days ago"
            elif delta.days < 30:
                weeks = delta.days // 7
                return f"{weeks} week{'s' if weeks > 1 else ''} ago"
            else:
                months = delta.days // 30
                return f"{months} month{'s' if months > 1 else ''} ago"

        hours = delta.seconds // 3600
        if hours > 0:
            return f"{hours} hour{'s' if hours > 1 else ''} ago"

        minutes = delta.seconds // 60
        if minutes > 0:
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"

        return "just now"
    except (ValueError, TypeError):
        return iso_timestamp
