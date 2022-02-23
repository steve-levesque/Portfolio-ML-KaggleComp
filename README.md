<!-- Repo's Banner -->
![Portfolio-ML-KaggleComp](https://user-images.githubusercontent.com/42849270/148702890-38dffea6-7303-41b4-b366-dc53c8d25694.png)



<!-- Shield Badges -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]



<!-- Description of the Project -->
## About

### Introduction
The Kaggle Competition Solver is a project aimed to offer a wrapper on Machine Learning models to reduce repetitive operations such as initializing the data, saving and its generic structure for similar models. In this way, multiples problems can run the same algorithm without recoding from scratch each time.

The state of efficacity is not very high, meaning data augmentation by example or variations in the data format can break the execution.



<!-- Repo's Content Tree -->
## Directories and Files
<details open>
  <summary><b>Project's Tree</b></summary>
    
  ``` bash
    |- data                   #
    |  |- logs                # A file log can be generated by the solver with a resume of the problem's data info.
    |  |- model_snapshots     # A model, depending of the module can be saved as a file for later use.
    |  |- predictions         # After the problem is solved, the predictions are generated here with sub_samples' structure.
    |  |- sub_samples         # Samples structure (Kaggle Competitions)
    |  |- test                # Test data goes here.
    |  \_ train               # Train data goes here.
    |- docs                   #
    |  |- plots               # Contains all plots made by the solver.
    |  \_ reports             # A report can be made about a problem solved in LaTeX.
    |     |- _report.sty      # Theme from NeurIPS 2021.
    |     \_ _template.tex    # Template of elements from NeurIPS 2021.
    |- models                 #
    |  |- ...                 # Multiples models, from sklearn, from scratch or others.
    |  \_ solver.py           # Wrapper class of the project.
    |- utils                  # Contains all utilities for the solver.
    |- .gitignore             #
    |- LICENSE                #
    |- README.md              # This file.
    \_ run_template.py        # Template to show structure.
  ```
</details>


<!-- Getting Started -->
## Installation
For this project to work, some programs needs to be installed with the required Python libraries:
- Python 3.x
- All the modules prompted to be install (i.e. the links in the later section below, or accept the prompt to download from your favorite IDE, if possible)

## How to Execute

1- Create a project at the root of this repositry.

2- Create a profile with your favorite IDE and specify a run_x.py file as the main.

3- Use the run_template and the other examples to use the solver with your own problem to solve.

4- Import your data in the same way as the example (MNIST), with a prefix that is the global sinificative name of the competiton or problem to solve.

5- Download link available below for comp1 and comp2 data.

<!-- Contribution -->
## Contribution

Contributions are always welcome, thank you for you time. Here are the steps to do so.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/MyContribution`)
3. Commit your Changes (`git commit -m 'Add MyContribution'`)
4. Push to the Branch (`git push origin feature/MyContribution`)
5. Open a Pull Request



<!-- License -->
## License

See the `LICENSE` file at the root of the project directory for more information.



<!-- Acknowlegements and Sources -->
## Acknowlegements and Sources
Sources
- *In each report, at the last section of each competition respectively. Ommitted here for simplicity.

Programs
- https://www.python.org/downloads/
- https://scikit-learn.org/
- http://hyperopt.github.io/hyperopt/
- https://keras.io/
- https://www.tensorflow.org/

Data Download Links
- Comp1 Train : https://storage.googleapis.com/stevelevesque.dev/ML/comp1_train.csv
- Comp1 Test : https://storage.googleapis.com/stevelevesque.dev/ML/comp1_test.csv
- Comp2 Train : https://storage.googleapis.com/stevelevesque.dev/ML/comp2_train.csv
- Comp2 Test : https://storage.googleapis.com/stevelevesque.dev/ML/comp2_test.csv

Authors
- Weiyue Cai



<!-- md links & imgs -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/steve-levesque/Portfolio-ML-KaggleComp.svg?style=for-the-badge
[contributors-url]: https://github.com/steve-levesque/Portfolio-ML-KaggleComp/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/steve-levesque/Portfolio-ML-KaggleComp.svg?style=for-the-badge
[forks-url]: https://github.com/steve-levesque/Portfolio-ML-KaggleComp/network/members
[stars-shield]: https://img.shields.io/github/stars/steve-levesque/Portfolio-ML-KaggleComp.svg?style=for-the-badge
[stars-url]: https://github.com/steve-levesque/Portfolio-ML-KaggleComp/stargazers
[issues-shield]: https://img.shields.io/github/issues/steve-levesque/Portfolio-ML-KaggleComp.svg?style=for-the-badge
[issues-url]: https://github.com/steve-levesque/Portfolio-ML-KaggleComp/issues
[license-shield]: https://img.shields.io/github/license/steve-levesque/Portfolio-ML-KaggleComp.svg?style=for-the-badge
[license-url]: https://github.com/steve-levesque/Portfolio-ML-KaggleComp/blob/main/LICENSE
