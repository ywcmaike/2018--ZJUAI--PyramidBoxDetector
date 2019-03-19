# pdpd输入格式的txt
pdfile = './final_our_Mall.txt'
# pytorch输入格式的txt
ptfile = './final_our_Mall_pt.txt'

f = open(pdfile, 'r')
f_pt = open(ptfile, 'w')
lines = f.readlines()
i = 0
rect = 0
total = 0
while i < len(lines):
    if 'jpg' in lines[i]:
        im_id = lines[i].rstrip()

        # print(im_id)
        num = int(lines[i + 1].rstrip())
        #
        i = i + 2
        box = []
        bad = 0

        for j in range(num):
            x1, y1, w, h = map(int, lines[i].rstrip().split(' ')[0:4])
            if w != h:
                print(im_id)
                print(w, h)
                rect += 1
            if w == 0 or h == 0:
                # print(im_id)
                bad += 1
                i = i + 1
                continue
            else:
                box.append([x1, y1, w, h])
                i = i + 1

        num = num - bad
        total += num
        if num > 0:
            f_pt.write(im_id)
            f_pt.write(' {0}'.format(num))
            for [x1, y1, w, h] in box:
                f_pt.write(' {0} {1} {2} {3}'.format(x1, y1, w, h))
            f_pt.write('\n')
        else:
            pass

    else:
        i = i + 1
f_pt.close()
f.close()
print(rect)
print(total)
