import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_cnn_lstm_model():
    fig, ax = plt.subplots(figsize=(15, 10))

    cnn_box = patches.FancyBboxPatch((0.1, 0.6), 0.3, 0.3, boxstyle="round,pad=0.05", edgecolor="black", facecolor="skyblue")
    ax.add_patch(cnn_box)
    ax.text(0.25, 0.75, 'Feature Extractor\n(CNN)', ha='center', va='center', fontsize=12, color='black')

    lstm_box = patches.FancyBboxPatch((0.5, 0.6), 0.3, 0.3, boxstyle="round,pad=0.05", edgecolor="black", facecolor="lightgreen")
    ax.add_patch(lstm_box)
    ax.text(0.65, 0.75, 'LSTM', ha='center', va='center', fontsize=12, color='black')

    fc_box = patches.FancyBboxPatch((0.8, 0.6), 0.15, 0.15, boxstyle="round,pad=0.05", edgecolor="black", facecolor="lightcoral")
    ax.add_patch(fc_box)
    ax.text(0.875, 0.675, 'Fully\nConnected', ha='center', va='center', fontsize=12, color='black')

    arrow_props = dict(facecolor='black', arrowstyle='->')
    ax.annotate('', xy=(0.4, 0.75), xytext=(0.35, 0.75), arrowprops=arrow_props)
    ax.annotate('', xy=(0.8, 0.75), xytext=(0.75, 0.75), arrowprops=arrow_props)

    ax.text(0.05, 0.75, 'Input\n(batch_size, seq_len, c, h, w)', ha='center', va='center', fontsize=12, color='black')
    ax.text(0.95, 0.675, 'Output\n(output_size)', ha='center', va='center', fontsize=12, color='black')

    ax.annotate('', xy=(0.4, 0.55), xytext=(0.35, 0.55), arrowprops=arrow_props)
    ax.annotate('', xy=(0.8, 0.55), xytext=(0.75, 0.55), arrowprops=arrow_props)
    ax.text(0.25, 0.5, 'Features\n(batch_size, seq_len, 1280)', ha='center', va='center', fontsize=12, color='black')
    ax.text(0.65, 0.5, 'Packed Features\n(batch_size, seq_len, hidden_size*2)', ha='center', va='center', fontsize=12, color='black')
    
    ax.annotate('', xy=(0.25, 0.65), xytext=(0.25, 0.6), arrowprops=arrow_props)
    ax.annotate('', xy=(0.65, 0.65), xytext=(0.65, 0.6), arrowprops=arrow_props)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.show()

draw_cnn_lstm_model()


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_cnn_lstm_model():
    fig, ax = plt.subplots(figsize=(15, 10))

    cnn_box = patches.FancyBboxPatch((0.1, 0.6), 0.3, 0.3, boxstyle="round,pad=0.05", edgecolor="black", facecolor="skyblue")
    ax.add_patch(cnn_box)
    ax.text(0.25, 0.75, 'Feature Extractor\n(CNN)', ha='center', va='center', fontsize=12, color='black')

    lstm_box = patches.FancyBboxPatch((0.5, 0.6), 0.3, 0.3, boxstyle="round,pad=0.05", edgecolor="black", facecolor="lightgreen")
    ax.add_patch(lstm_box)
    ax.text(0.65, 0.75, 'LSTM', ha='center', va='center', fontsize=12, color='black')

    fc_box = patches.FancyBboxPatch((0.8, 0.6), 0.15, 0.15, boxstyle="round,pad=0.05", edgecolor="black", facecolor="lightcoral")
    ax.add_patch(fc_box)
    ax.text(0.875, 0.69, 'Fully\nConnected', ha='center', va='center', fontsize=12, color='black')
    # ax.text(0.875, 0.66, '', ha='center', va='center', fontsize=12, color='black')

    arrow_props = dict(facecolor='black', arrowstyle='->')
    ax.annotate('', xy=(0.4, 0.75), xytext=(0.35, 0.75), arrowprops=arrow_props)
    ax.annotate('', xy=(0.8, 0.75), xytext=(0.75, 0.75), arrowprops=arrow_props)

    ax.text(0.05, 0.75, 'Input\n(batch_size, seq_len, c, h, w)', ha='center', va='center', fontsize=12, color='black')
    ax.text(0.95, 0.65, 'Output\n(output_size)', ha='center', va='center', fontsize=12, color='black')

    ax.annotate('', xy=(0.4, 0.55), xytext=(0.35, 0.55), arrowprops=arrow_props)
    ax.annotate('', xy=(0.8, 0.55), xytext=(0.75, 0.55), arrowprops=arrow_props)
    ax.text(0.25, 0.5, 'Features\n(batch_size, seq_len, 1280)', ha='center', va='center', fontsize=12, color='black')
    ax.text(0.65, 0.5, 'Packed Features\n(batch_size, seq_len, hidden_size*2)', ha='center', va='center', fontsize=12, color='black')
    
    ax.annotate('', xy=(0.25, 0.65), xytext=(0.25, 0.6), arrowprops=arrow_props)
    ax.annotate('', xy=(0.65, 0.65), xytext=(0.65, 0.6), arrowprops=arrow_props)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.show()

draw_cnn_lstm_model()


# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# def draw_cnn_lstm_model_diagram():
#     fig, ax = plt.subplots(figsize=(16, 8))

#     # Colors for blocks
#     colors = {
#         "input": "#FFDDC1",
#         "cnn": "#A2D2FF",
#         "lstm": "#B9FBC0",
#         "fc": "#FFC4C4",
#         "output": "#FFABAB"
#     }

#     # Input block
#     input_box = patches.FancyBboxPatch((0.05, 0.5), 0.2, 0.15, boxstyle="round,pad=0.1", edgecolor="black", facecolor=colors["input"])
#     ax.add_patch(input_box)
#     ax.text(0.15, 0.575, "Input\n(batch_size, seq_len, c, h, w)", ha="center", va="center", fontsize=10, color="black")

#     # CNN block
#     cnn_box = patches.FancyBboxPatch((0.3, 0.5), 0.25, 0.15, boxstyle="round,pad=0.1", edgecolor="black", facecolor=colors["cnn"])
#     ax.add_patch(cnn_box)
#     ax.text(0.425, 0.575, "Feature Extractor\n(CNN)", ha="center", va="center", fontsize=10, color="black")
#     ax.text(0.425, 0.525, "Output: (batch_size, seq_len, 1280)", ha="center", va="center", fontsize=9, color="black")

#     # LSTM block
#     lstm_box = patches.FancyBboxPatch((0.6, 0.5), 0.25, 0.15, boxstyle="round,pad=0.1", edgecolor="black", facecolor=colors["lstm"])
#     ax.add_patch(lstm_box)
#     ax.text(0.725, 0.575, "LSTM\n(Temporal Analysis)", ha="center", va="center", fontsize=10, color="black")
#     ax.text(0.725, 0.525, "Output: (batch_size, seq_len, hidden_size*2)", ha="center", va="center", fontsize=9, color="black")

#     # Fully Connected block
#     fc_box = patches.FancyBboxPatch((0.9, 0.5), 0.2, 0.15, boxstyle="round,pad=0.1", edgecolor="black", facecolor=colors["fc"])
#     ax.add_patch(fc_box)
#     ax.text(1.0, 0.575, "Fully Connected\n(Output)", ha="center", va="center", fontsize=10, color="black")
#     ax.text(1.0, 0.525, "Output: (batch_size, seq_len, output_size)", ha="center", va="center", fontsize=9, color="black")

#     # Arrows
#     arrow_props = dict(facecolor="black", arrowstyle="->", linewidth=2)
#     ax.annotate("", xy=(0.25, 0.575), xytext=(0.225, 0.575), arrowprops=arrow_props)  # Input to CNN
#     ax.annotate("", xy=(0.55, 0.575), xytext=(0.525, 0.575), arrowprops=arrow_props)  # CNN to LSTM
#     ax.annotate("", xy=(0.85, 0.575), xytext=(0.825, 0.575), arrowprops=arrow_props)  # LSTM to Fully Connected

#     # Descriptive labels
#     ax.text(0.15, 0.65, "Step 1: Input Data", fontsize=10, ha="center", va="center", color="black")
#     ax.text(0.425, 0.65, "Step 2: Feature Extraction", fontsize=10, ha="center", va="center", color="black")
#     ax.text(0.725, 0.65, "Step 3: Temporal Analysis", fontsize=10, ha="center", va="center", color="black")
#     ax.text(1.0, 0.65, "Step 4: Final Prediction", fontsize=10, ha="center", va="center", color="black")

#     # Adjust layout and remove axes
#     ax.set_xlim(0, 1.2)
#     ax.set_ylim(0, 1)
#     ax.axis("off")

#     plt.show()

# draw_cnn_lstm_model_diagram()

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_yolo_lstm_model():
    fig, ax = plt.subplots(figsize=(15, 10))

    # YOLOv7-tiny Feature Extractor
    yolo_box = patches.FancyBboxPatch((0.1, 0.6), 0.3, 0.3, boxstyle="round,pad=0.05", edgecolor="black", facecolor="skyblue")
    ax.add_patch(yolo_box)
    ax.text(0.25, 0.75, 'YOLOv7-tiny\nFeature Extractor', ha='center', va='center', fontsize=12, color='black')

    # LSTM Model
    lstm_box = patches.FancyBboxPatch((0.5, 0.6), 0.3, 0.3, boxstyle="round,pad=0.05", edgecolor="black", facecolor="lightgreen")
    ax.add_patch(lstm_box)
    ax.text(0.65, 0.75, 'LSTM\n(Temporal Analysis)', ha='center', va='center', fontsize=12, color='black')

    # Fully Connected Layer
    fc_box = patches.FancyBboxPatch((0.8, 0.6), 0.15, 0.15, boxstyle="round,pad=0.05", edgecolor="black", facecolor="lightcoral")
    ax.add_patch(fc_box)
    ax.text(0.875, 0.675, 'Fully\nConnected', ha='center', va='center', fontsize=12, color='black')

    # Arrows to show the flow
    arrow_props = dict(facecolor='black', arrowstyle='->')
    ax.annotate('', xy=(0.4, 0.75), xytext=(0.35, 0.75), arrowprops=arrow_props)
    ax.annotate('', xy=(0.8, 0.75), xytext=(0.75, 0.75), arrowprops=arrow_props)

    # Input and Output Labels
    ax.text(0.05, 0.75, 'Input\n(batch_size, seq_len, c, h, w)', ha='center', va='center', fontsize=12, color='black')
    ax.text(0.95, 0.675, 'Output\n(output_size)', ha='center', va='center', fontsize=12, color='black')

    # Intermediate Representations
    ax.annotate('', xy=(0.4, 0.55), xytext=(0.35, 0.55), arrowprops=arrow_props)
    ax.annotate('', xy=(0.8, 0.55), xytext=(0.75, 0.55), arrowprops=arrow_props)
    ax.text(0.25, 0.5, 'Feature Maps\n(batch_size, seq_len, 1024)', ha='center', va='center', fontsize=12, color='black')
    ax.text(0.65, 0.5, 'Temporal Features\n(batch_size, seq_len, hidden_size*2)', ha='center', va='center', fontsize=12, color='black')
    
    ax.annotate('', xy=(0.25, 0.65), xytext=(0.25, 0.6), arrowprops=arrow_props)
    ax.annotate('', xy=(0.65, 0.65), xytext=(0.65, 0.6), arrowprops=arrow_props)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.show()

# Draw the model structure
draw_yolo_lstm_model()













