import numpy as np
import matplotlib.pyplot as plt


def grid_generator(nx, ny):
    obstacle_map = np.zeros((ny, nx), dtype=int)
    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(obstacle_map, cmap='Reds', vmin=0,
                   vmax=1, interpolation='nearest')

    # Add 'last_pos' to prevent toggling the same cell multiple times on mouse jitter
    state = {'drawing': False, 'dragged': False, 'last_pos': None}

    def on_press(event):
        """Handler for mouse button press events."""
        if event.inaxes != ax:
            return
        state['drawing'] = True
        state['dragged'] = False
        state['last_pos'] = None  # Reset last position on new click

    def on_release(event):
        """Handler for mouse button release events."""
        if event.inaxes != ax:
            state['drawing'] = False
            return

        if state['drawing'] and not state['dragged']:
            ix, iy = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= ix < nx and 0 <= iy < ny:
                obstacle_map[iy, ix] = 1 - obstacle_map[iy, ix]
                im.set_data(obstacle_map)
                fig.canvas.draw_idle()

        state['drawing'] = False

    def on_motion(event):
        """Handler for mouse motion events."""
        if not state['drawing'] or event.inaxes != ax:
            return

        state['dragged'] = True
        ix, iy = int(round(event.xdata)), int(round(event.ydata))

        # Check if mouse is within bounds and if it has moved to a new cell
        if 0 <= ix < nx and 0 <= iy < ny and (ix, iy) != state['last_pos']:
            # --- BEHAVIOR CHANGE IS HERE ---
            # Now, dragging always toggles the cell's state (0->1, 1->0)
            # The mouse button used for dragging no longer matters.
            obstacle_map[iy, ix] = 1 - obstacle_map[iy, ix]
            im.set_data(obstacle_map)
            fig.canvas.draw_idle()
            state['last_pos'] = (ix, iy)  # Update the last processed position

    # Plot setup (executed only once)
    ax.set_xticks(np.arange(-.5, nx, 1), minor=True)
    ax.set_yticks(np.arange(-.5, ny, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks(np.arange(0, nx, 1))
    ax.set_yticks(np.arange(0, ny, 1))
    # Updated title to reflect the new, simpler behavior
    ax.set_title("Click or Drag to Toggle Obstacles")

    # Connect the event handlers
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    plt.show()
    return obstacle_map
