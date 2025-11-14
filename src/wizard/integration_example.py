"""
Example integration of progress tracking into training

This shows how trainers should integrate with the progress tracking system.
"""

from .progress_tracker import ProgressTracker


class TrainingIntegration:
    """
    Example showing how to integrate progress tracking into a trainer.

    Trainers should:
    1. Start a training session when training begins
    2. Update progress periodically during training
    3. Complete the session when done, registering the model
    4. Handle failures gracefully
    """

    @staticmethod
    def sft_training_example():
        """
        Example SFT training integration
        """

        # Initialize tracker
        tracker = ProgressTracker()

        # Training configuration
        config = {
            "model": "gpt2",
            "dataset": "yahma/alpaca-cleaned",
            "batch_size": 4,
            "epochs": 3,
            "learning_rate": 2e-5
        }

        # Start training session
        session_id = tracker.start_training_session(
            method="sft",
            config=config,
            total_steps=1000  # Total training steps
        )

        try:
            # Simulate training loop
            for step in range(1000):
                # ... actual training code ...

                # Update progress every 10 steps
                if step % 10 == 0:
                    tracker.update_training_progress(
                        session_id=session_id,
                        current_step=step,
                        metrics={"loss": 1.234}  # Current metrics
                    )

            # Training completed successfully
            checkpoint_path = "checkpoints/sft/my_model"
            model_id = "gpt2_alpaca_v1"

            # Register the model in the registry
            tracker.registry.register_model(
                model_id=model_id,
                base_model="gpt2",
                training_id=session_id,
                method="sft",
                checkpoint_path=checkpoint_path,
                dataset="yahma/alpaca-cleaned",
                tags=["instruction-following", "alpaca"],
                notes="First SFT training run",
                metrics={"final_loss": 1.123, "perplexity": 3.45}
            )

            # Complete the training session
            tracker.complete_training_session(
                session_id=session_id,
                checkpoint_path=checkpoint_path,
                final_metrics={"final_loss": 1.123, "perplexity": 3.45}
            )

            print(f"Training complete! Model registered as: {model_id}")

        except Exception as e:
            # Handle training failure
            tracker.fail_training_session(session_id, str(e))
            raise

    @staticmethod
    def dpo_training_example():
        """
        Example DPO training integration (requires SFT model)
        """

        tracker = ProgressTracker()

        # Check prerequisites
        prereqs = tracker.prereq_checker.check_dpo_prerequisites()

        if not prereqs.satisfied:
            print("Cannot start DPO:")
            print(prereqs.message)
            for suggestion in prereqs.suggestions:
                print(f"  - {suggestion}")
            return

        # Use the suggested SFT model
        sft_model = prereqs.available_models[0]
        print(f"Using SFT model: {sft_model.id}")

        # Create a new model ID for DPO version
        model_id = f"{sft_model.id}_dpo"

        # Mark model as training (adds to lineage)
        tracker.registry.mark_model_training(
            model_id=model_id if model_id in tracker.registry.state["models"] else sft_model.id,
            training_id=f"dpo_{int(__import__('time').time())}",
            method="dpo",
            checkpoint_path=f"checkpoints/dpo/{model_id}",
            dataset="Anthropic/hh-rlhf",
            total_steps=500
        )

        # ... training code ...

        # After training, register as new model
        session_id = f"dpo_{int(__import__('time').time())}"

        tracker.registry.register_model(
            model_id=model_id,
            base_model=sft_model.base_model,
            training_id=session_id,
            method="dpo",
            checkpoint_path=f"checkpoints/dpo/{model_id}",
            dataset="Anthropic/hh-rlhf",
            parent_model_id=sft_model.id,
            tags=["preference-aligned", "dpo"],
            notes="DPO-tuned for helpfulness and harmlessness"
        )

    @staticmethod
    def pipeline_training_example():
        """
        Example using pipelines for multi-stage training
        """

        tracker = ProgressTracker()

        # Import pipeline manager
        from .pipelines import PipelineManager

        pipeline_mgr = PipelineManager(tracker.registry)

        # Create pipeline from template
        pipeline = pipeline_mgr.create_from_template(
            template_name="dpo_simple",
            base_model="gpt2"
        )

        print(f"Created pipeline: {pipeline.name}")
        print(f"Stages: {' → '.join([s.stage for s in pipeline.stages])}")

        # Execute pipeline stages
        while True:
            next_stage = pipeline_mgr.get_next_stage(pipeline.id)
            if not next_stage:
                print("Pipeline complete!")
                break

            stage_idx, stage = next_stage
            print(f"Starting stage: {stage.stage}")

            # Here you would launch the appropriate training wizard/function
            # For example:
            # if stage.stage == "sft":
            #     model_id = train_sft(...)
            #     pipeline_mgr.complete_stage(pipeline.id, stage_idx, model_id)
            # elif stage.stage == "dpo":
            #     model_id = train_dpo(...)
            #     pipeline_mgr.complete_stage(pipeline.id, stage_idx, model_id)

            break  # For example purposes


# Command-line testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python integration_example.py [sft|dpo|pipeline]")
        sys.exit(1)

    example_type = sys.argv[1]

    if example_type == "sft":
        TrainingIntegration.sft_training_example()
    elif example_type == "dpo":
        TrainingIntegration.dpo_training_example()
    elif example_type == "pipeline":
        TrainingIntegration.pipeline_training_example()
    else:
        print(f"Unknown example type: {example_type}")
