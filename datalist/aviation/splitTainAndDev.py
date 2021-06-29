
def processSyllable():
    syllable_file = open("syllable.txt", "r", encoding="utf-8")
    syllable_txt = syllable_file.readlines()
    syllable_file.close()

    train_syllable_file = open("train.syllable.txt", "w", encoding="utf-8")
    dev_syllable_file = open("dev.syllable.txt", "w", encoding="utf-8")
    for line in syllable_txt:
        line = line.split("\t")
        if int(line[0].split("_")[1]) <= 10:
            dev_syllable_file.writelines(line[0] + "\t" + line[1])
        else:
            train_syllable_file.writelines(line[0] + "\t" + line[1])
    train_syllable_file.close()
    dev_syllable_file.close()

def processWavLst():
    syllable_file = open("wav.lst", "r", encoding="utf-8")
    syllable_txt = syllable_file.readlines()
    syllable_file.close()

    train_syllable_file = open("train.wav.lst", "w", encoding="utf-8")
    dev_syllable_file = open("dev.wav.lst", "w", encoding="utf-8")
    for line in syllable_txt:
        line = line.split("\t")
        if int(line[0].split("_")[1]) <= 10:
            dev_syllable_file.writelines(line[0] + "\t" + line[1])
        else:
            train_syllable_file.writelines(line[0] + "\t" + line[1])
    train_syllable_file.close()
    dev_syllable_file.close()

if __name__ == "__main__":
    processSyllable()
    processWavLst()