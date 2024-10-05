import os

def SupprimeLineVide(labels_dir):
            labels = os.listdir(labels_dir)
            for label in labels:
                label_path = os.path.join(labels_dir, label)
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    non_empty_lines = [line.strip() for line in lines if line.strip()]

                with open(label_path, 'w') as f:
                    f.write('\n'.join(non_empty_lines))
                print(f'{label} handled')







