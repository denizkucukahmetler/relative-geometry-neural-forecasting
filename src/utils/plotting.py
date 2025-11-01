from matplotlib import pyplot as plt


def plot_trajectories_3d(predicted, ground_truth, batch_index=None):
    # Select the batch
    if batch_index:
        pred = predicted[batch_index].cpu().detach().numpy()
        truth = ground_truth[batch_index].cpu().detach().numpy()
    else:
        pred = predicted.cpu().detach().numpy()
        truth = ground_truth.cpu().detach().numpy()

    # Extract x, y, z for both predicted and ground truth
    pred_x, pred_y, pred_z = pred[:, 0], pred[:, 1], pred[:, 2]
    truth_x, truth_y, truth_z = truth[:, 0], truth[:, 1], truth[:, 2]

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground truth
    ax.plot(truth_x, truth_y, truth_z, label='Ground Truth', color='blue', linewidth=2)

    # Plot predicted trajectory
    ax.plot(pred_x, pred_y, pred_z, label='Predicted', color='red', linestyle='--', linewidth=2)

    # Add labels and legend
    ax.set_title('Predicted vs Ground Truth Trajectories')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()
