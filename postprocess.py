

sentences = []
with open("data/test.txt", "r") as f:
    for line in f:
        sentences.append(len(line.strip().split(" ")))

tags = []
with open("test_pred.txt", "r") as f:
    for line in f:
        tags.append(line.strip().split(" "))

for i in range(len(sentences)):
    if sentences[i] != len(tags[i]):
        tags[i] = tags[i][:sentences[i]]

with open("test_pred_final.txt", "w") as f:
    for tag in tags:
        f.write(" ".join(tag) + "\n")
