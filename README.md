<!-- ABOUT THE PROJECT -->

## About The Project

TEDISA (Tensor Dimensions Static Analyzer) is a static analysis tool designed to analyze tensor dimension changes in PyTorch and NumPy programs, specifically focusing on Neural Networks applications.

## Input Specifications

The analyzer accepts two types of inputs:

1. Direct path to a source file containing PyTorch operations
2. Directory path for recursive analysis of Python source files (`.py` or `.ipynb`)

## Output Format

The analysis generates detailed reports in three available formats:

- Plain text (`.txt`) - For easy reading and parsing
- PDF (`.pdf`) - For formal documentation
- HTML (`.html`) - For interactive visualization

1. Tensor operations affecting dimensions
2. Operation frequency statistics
3. Detailed operation analysis including:
   - Operation name
   - Line number
   - Initial and final dimensions
   - Operation identifier/context
   - Usage description

Example of text output format:

```txt
Tensor A | Line  97 | op: reshape     | dim 2 -> dim 5  | use: before loading
Tensor A | Line  98 | op: flatten     | dim 5 -> dim 2  | use: to next batch
Tensor B | Line 112 | op: unsqueeze   | dim 3 -> dim 4  | use: for broadcasting
Tensor C | Line  45 | op: permute     | dim 4 -> dim 4  | use: channel last
```

## Analysis Strategy

Technical Approach:

- Static analysis through Abstract Syntax Tree (AST) traversal
- Tracking tensor declarations and their dimensional changes

This tool aims to provide insights into tensor dimension manipulation patterns in real-world applications without requiring code execution.

### Prerequisites

```
under construction
```

## Usage

Basic usage of TEDISA through command line:

```bash
# Analyze a single Python file
python tedisa.py file.py -o output_directory

# Analyze a single file with specific output format (txt, pdf, or html)
python tedisa.py file.py -o output_directory --format txt
python tedisa.py file.py -o output_directory --format pdf
python tedisa.py file.py -o output_directory --format html

# Analyze all Python files in a directory
python tedisa.py /path/to/directory -o output_directory

# Analyze with recursive directory search
python tedisa.py /path/to/directory -o output_directory --recursive
```

The output will be generated in the specified directory with the following structure:

```
under construction
```

## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->

## Contact

email: jcrucesdeveloper@gmail.com

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
