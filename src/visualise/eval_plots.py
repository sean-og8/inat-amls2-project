def create_eval_plots(eval_metrics, model_name):
    # Convert eval_metrics to a DataFrame and add an 'epoch' column
    plot_df = pd.DataFrame.from_dict(eval_metrics)
    plot_df["epoch"] = plot_df.index + 1

    # Loss Plot 
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(plot_df["epoch"], plot_df["train_loss"], label="Training Loss",
            color="darkblue", linewidth=2)
    ax.plot(plot_df["epoch"], plot_df["val_loss"], label="Validation Loss",
            color="darkgreen", linewidth=2)
    
    # styling
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.7)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=12)
    # save plot
    fig.savefig(f"outputs/loss_{model_name}.png", bbox_inches="tight")
    
    # Accuracy Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scale accuracy values to percentages
    plot_df["train_acc"] *= 100
    plot_df["val_acc"] *= 100
    
    # Plot training and validation accuracies
    ax.plot(plot_df["epoch"], plot_df["train_acc"], label="Training Accuracy",
            color="darkblue", linewidth=2)
    ax.plot(plot_df["epoch"], plot_df["val_acc"], label="Validation Accuracy",
            color="darkgreen", linewidth=2)
    
    # styling
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.7)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=12)
    # save plot
    fig.savefig(f"outputs/accuracy_{model_name}.png", bbox_inches="tight")
    plt.close("all")
    return