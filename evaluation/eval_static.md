# Static Evaluation Results

## ROUGE-L Summary Table

| Prompt | Base ROUGE-L | LoRA ROUGE-L |
| ------ | ------------ | ------------ |
| 1      | 0.165        | 0.203        |
| 2      | 0.115        | 0.135        |
| 3      | 0.000        | 0.028        |
| 4      | 0.192        | 0.133        |
| 5      | 0.049        | 0.069        |
| 6      | 0.061        | 0.086        |
| 7      | 0.103        | 0.141        |

---

## Prompt 1

**Prompt:** Create a new Git branch and switch to it.
**Reference:**

```
1. git checkout -b <branch-name>
2. git switch <branch-name>
```

**Base Model Output:**

````
1. Create a new Git branch:

   ```bash
   git checkout -b feature/my-new-feature
````

2. Switch to the new branch:

   ```bash
   git checkout feature/my-new-feature
   ```

3. Commit your changes:

   ```bash
   git add .
   git commit -m "Added feature"
   ```

4. Push your changes to the remote repository:

   ```bash
   git push origin feature/my-new-feature
   ```

5. Merge the changes into the master branch:

   ```bash
   git checkout master
   git
   ```

```
**Fine-tuned Model Output:**
```

To create a new Git branch:

1. Open the Git command line interface (CLI) or terminal on your computer.
2. Type `git branch` to see all the available branches.
3. Choose the name of the new branch you want to create, and press Enter.
4. The new branch will be created with a default name like "feature-branch" in this example.

Switching to a new branch:

1. Open the Git command line interface (CLI) or terminal on your computer.
2. Type `git checkout -b feature-branch` to switch to the new branch.
3. The current branch will be renamed to "feature-branch".

```
| Model | ROUGE-L |
|-------|---------|
| Base  | 0.165 |
| LoRA  | 0.203 |

---

## Prompt 2
**Prompt:** Compress the folder reports into reports.tar.gz.
**Reference:**
```

tar -czvf reports.tar.gz reports/

```
**Base Model Output:**
```

1. Run `tar` command to create a compressed archive of the folder reports.
2. Use `tar czf` command to compress the output file (`reports.tar.gz`) into a new archive named `reports.tar.gz`.
3. Move the newly created `reports.tar.gz` file to its original location.

4. You can check if the archive was successfully created by checking the contents of the archive. If it contains the required files, then your tar command was successful in compressing them.

```
**Fine-tuned Model Output:**
```

1. First, create a new directory named reports and move the folder reports inside it.

2. Next, open your terminal or command prompt and run the following command to tar up the folder reports:

```
cd reports
tar -czvf reports.tar.gz reports
```

3. Finally, move the compressed reports.tar.gz file to another location on your computer (e.g., C:\reports) to access them easily in the future.

Note: You can also specify additional options such as `-d` (to create a single directory), `-z`, and `-cv` (to use compression level `z`) for the `tar` command.

```
| Model | ROUGE-L |
|-------|---------|
| Base  | 0.115 |
| LoRA  | 0.135 |

---

## Prompt 3
**Prompt:** List all Python files in the current directory recursively.
**Reference:**
```

find . -name '\*.py'

```
**Base Model Output:**
```


```

```
**Fine-tuned Model Output:**
```

```python
import os

def list_recursive(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            print(os.path.join(root, file))

list_recursive(".")
```

```
| Model | ROUGE-L |
|-------|---------|
| Base  | 0.000 |
| LoRA  | 0.028 |

---

## Prompt 4
**Prompt:** Set up a virtual environment and install requests.
**Reference:**
```

python3 -m venv venv
source venv/bin/activate
pip install requests

```
**Base Model Output:**
```

1. Create a new virtual environment in your terminal or command prompt by running the following command:

```
python -m venv my_env
```

2. Activate the virtual environment by running the following command:

```
source my_env/bin/activate
```

3. Install requests using pip by running the following command:

```
pip install requests
```

4. Test that the installation was successful by running the following command:

```
python -m requests.testing
```

5. You should now be able to use `requests` library in your Python code.

```
**Fine-tuned Model Output:**
```

1. Open your terminal or command prompt
2. Create a new directory for the project using the `mkdir` command, then navigate to that directory using the `cd` command
3. Change into the directory you just created by entering `cd <directory name>`
4. Use the `pip` package manager to install requests using the following command:
   ```
   pip install requests
   ```
5. Once the installation is complete, you can test if it works by running the following code in your Python script:

```python
import requests

# Replace 'your-url' with the actual URL of your website
response = requests.get('https://example.com')
print(
```

| Model | ROUGE-L |
| ----- | ------- |
| Base  | 0.192   |
| LoRA  | 0.133   |

---

## Prompt 5

**Prompt:** Fetch only the first ten lines of a file named output.log.
**Reference:**

```
head -n 10 output.log
```

**Base Model Output:**

````
To fetch the first ten lines of a file named output.log, we use the `head` command with the `--lines=10` option to specify how many lines to retrieve. Here's the modified shell script:

```bash
#!/bin/bash

while read -r line; do
    # Perform your desired action on each line
done < /path/to/output.log | head -n 10
````

This script reads the contents of the specified file (`/path/to/output.log`) and pipes it through the `head` command with the `-n 10` option (which limits the number of lines to ten), and outputs them to

```
**Fine-tuned Model Output:**
```

To fetch only the first ten lines of an existing file, you can use the `head` command along with the `-n` option to specify the number of lines you want to retrieve.

Example:

```
$ head -n10 log.txt
```

This will retrieve and display the first ten lines from the log.txt file.

```
| Model | ROUGE-L |
|-------|---------|
| Base  | 0.049 |
| LoRA  | 0.069 |

---

## Prompt 6
**Prompt:** Remove all .pyc files but keep the .py files intact
**Reference:**
```

find . -name '\*.pyc' -delete

```
**Base Model Output:**
```

1. Use `find` command with `-name` option to search for `.pyc` files.
2. Use `grep` command with the pattern `\.pyc$` to remove all `.pyc` files.
3. Use `mv` command with the output of `find` and the input of `grep` to move only the `.py` files.

Example:

```
#!/bin/bash

# Find all .pyc files in a directory
echo "Find all .pyc files..."
find . -type f \( -name "*.pyc" -o -name "*.py" \) | grep '\.pyc$' > /dev/null
```

**Fine-tuned Model Output:**

```
1. Using Python's built-in function `os.listdir()` to list all the directories in a specified directory, and using `find` command to find all `.pyc` files in each directory.

2. Using the `shutil` module to copy only the `.py` files from each subdirectory to another directory named `clean_folder`, which will be deleted after running the script.

3. Using the `os` module to create a new directory with the same name as `clean_folder`, and recursively delete all `.pyc` files found in it.

4. Finally, deleting the newly created `clean_folder` directory along with its contents.

I
```

| Model | ROUGE-L |
| ----- | ------- |
| Base  | 0.061   |
| LoRA  | 0.086   |

---

## Prompt 7

**Prompt:** Find all files larger than 100MB in the current directory and its subdirectories, then sort them by size
**Reference:**

```
find . -type f -size +100M -exec ls -lh {} + | sort -k 5 -rh
```

**Base Model Output:**

````
To find all files larger than 100MB in the current directory and its subdirectories, we can use a combination of the `find` command and the `size` option to specify the maximum file size. Here's an example script that does exactly that:

```sh
#!/bin/bash

# Get current directory path
cd "$(pwd)"

# List all files and directories larger than 100MB (in bytes)
files=$(find . -type f -mtime +60 -size +100M -exec ls {} \; | awk '{print $9}' | sort -nr | head -n 50)

# Print
````

**Fine-tuned Model Output:**

````
Here's a Bash script that finds all files larger than 100MB in the current directory and its subdirectories. It first sorts them alphabetically by name. Then it filters out any files with an extension of .DS_Store (since it's not relevant to our purposes). Finally, it loops through each file, displaying its name and size as well as its path and modification date. Here's the script:

```bash
#!/bin/bash

# Sort files by their names alphabetically
sort -g -n -o "$(dirname "$0")"/*.{bin,conf,data,log,mnt,opt,run,sys,var}*" |
````

| Model | ROUGE-L |
| ----- | ------- |
| Base  | 0.103   |
| LoRA  | 0.141   |

---
