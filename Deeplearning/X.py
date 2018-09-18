sequences = {}
seqs_name = []
Ac_comparison_table = {'A': '1', 'V': '2', 'L': '3', 'I': '4', 'F': '5', 'W': '6', 'M': '7', 'P': '8', 'G': '9',
                       'S': '10', 'T': '11', 'C': '12', 'Y': '13', 'N': '14', 'Q': '15', 'H': '16', 'K': '17',
                       'R': '18', 'D': '19', 'E': '20'}
with open('dabase_test/seqs_mapping.fas', mode='r+') as file:
    file = file.read().split('\n')
    for line in range(len(file)):
        if '>' in file[line]:
            value = file[line]
            value = value.replace('>', '')
            name = value
            seqs_name.append(name)
            sequence = ''
            for i in range(1, 7):
                line = line + 1
                sequence = sequence + file[line]
            sequence = sequence.replace(' ', '')
            sequences[name] = sequence
    final = len(seqs_name) - 1


def genetic_mutation(file, name1, name2, seq1, seq2):
    #file.write(name1 + '与' + name2 + '的比对结果为：'+'\n')
    for num in range(len(seq1)):
        if num == 328:
            file.write(Ac_comparison_table[seq1[num]])
        elif seq1[num] == seq2[num]:
            file.write(Ac_comparison_table[seq1[num]] + ',')
        elif seq1[num] != seq2[num]:
            row = int(Ac_comparison_table[seq1[num]]) - 1
            column = int(Ac_comparison_table[seq2[num]]) - 1
            mid = row*20+column+30
            if mid <= 50:
                file.write(str(mid) + ',')
            elif 51 < mid <= 71:
                file.write(str(mid-1) + ',')
            elif 72 < mid <= 92:
                file.write(str(mid-2) + ',')
            elif 92 < mid <= 113:
                file.write(str(mid-3) + ',')
            elif 113 < mid <= 134:
                file.write(str(mid-4) + ',')
            elif 134 < mid <= 155:
                file.write(str(mid-5) + ',')
            elif 155 < mid <= 176:
                file.write(str(mid-6) + ',')
            elif 176 < mid <= 197:
                file.write(str(mid-7) + ',')
            elif 197 < mid <= 218:
                file.write(str(mid-8) + ',')
            elif 218 < mid <= 239:
                file.write(str(mid-9) + ',')
            elif 239 < mid <= 260:
                file.write(str(mid-10) + ',')
            elif 260 < mid <= 281:
                file.write(str(mid-11) + ',')
            elif 281 < mid <= 302:
                file.write(str(mid-12) + ',')
            elif 302 < mid <= 323:
                file.write(str(mid-13) + ',')
            elif 323 < mid <= 344:
                file.write(str(mid-14) + ',')
            elif 344 < mid <= 365:
                file.write(str(mid-15) + ',')
            elif 365 < mid <= 386:
                file.write(str(mid-16) + ',')
            elif 386 < mid <= 407:
                file.write(str(mid-17) + ',')
            elif 407 < mid <= 409:
                file.write(str(mid-18) + ',')
    file.write('\n')


filename = 'result2.txt'
with open('dabase_test/' + filename, 'w') as file:
    for seq in range(len(seqs_name)):
        seq1 = seq
        seq2 = 0
        while seq2 <= final:
            if seq1 == seq2:
                seq2 = seq2 + 1
                continue
            else:
                a = sequences[seqs_name[seq1]]
                b = sequences[seqs_name[seq2]]
                genetic_mutation(file, seqs_name[seq1], seqs_name[seq2], a, b)
                seq2 = seq2 + 1
