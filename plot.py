import matplotlib.pyplot as plt

def make_plots(plot_list, plot_info):
    plt.figure(figsize=(10, 6))
    plt.title(f"{plot_info} Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("AUC/loss")
    plot_list = list(map(lambda x: x * 100, plot_list))
    # plt.yticks(plot_list)
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.plot(range(1, len(plot_list) + 1), plot_list, label=plot_info, marker='o', linestyle='-')
    plt.legend()

    # plt.ylim(min(plot_list) - 0.001, max(plot_list) + 0.001)
    plt.grid(True)
    
    # plt.tight_layout()
    plt.savefig(f"{plot_info}_{0.00125}.png")
    plt.close()

train_loss_values = [item["train_loss"] for item in data]
train_ce_loss_values = [item["train_ce_loss"] for item in data]
train_tn_loss_values = [item["train_tn_loss"] for item in data]
test_loss_values = [item["test_loss"] for item in data]
test_auc_values = [item["test_auc_avg"] for item in data]

make_plots(train_loss_values, 'train_loss')
make_plots(test_loss_values[1:], 'test_loss')
make_plots(train_ce_loss_values, 'ce_loss')
make_plots(train_tn_loss_values, 'tn_loss')
make_plots(test_auc_values[1:], 'auc')

