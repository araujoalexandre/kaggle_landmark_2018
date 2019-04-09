
from os.path import join
import sys
import glob

def run():
    if len(sys.argv) != 3:
        print('Syntax: {} <input_dir/> <list_path>'.format(sys.argv[0]))
        sys.exit(0)
    (input_dir, list_path) = sys.argv[1:]

    count = 0
    with open(list_path, 'w') as f:
        for image_path in glob.iglob(join(input_dir, '*')):
            f.write('{}\n'.format(image_path))
            count += 1
    print(count)

if __name__ == '__main__':
    run()