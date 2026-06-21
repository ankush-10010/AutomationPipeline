with open(r'C:\CODE\automation\ai_explainer\notebooks\orchestrator_kaggle.ipynb', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('Kaggle', 'Colab')
text = text.replace('kaggle', 'colab')
text = text.replace('/colab/input', '/content')
text = text.replace('/colab/working', '/content')
text = text.replace('T4 x2 or P100', 'T4 GPU')
text = text.replace('\'Output\' tab on the right side', '\'Files\' tab on the left side')

with open(r'C:\CODE\automation\ai_explainer\notebooks\orchestrator_colab.ipynb', 'w', encoding='utf-8') as f:
    f.write(text)
    