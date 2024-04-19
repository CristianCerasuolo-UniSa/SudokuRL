# SudokuRL

SudokuRL is an open source project that aims to solve the Sudoku game using Reinforcement Learning techniques. The project was started with the aim of exploring the potential of machine learning in solving complex problems such as Sudoku.

## Actual State

Currently the system works, the agent trains using Deep Q-Learning on a very diverse sudoku database. The agent is absolutely unable to solve any sudoku as he prefers to violate the rules rather than respect them. During the execution of the code, some graphs are saved that show the progress of the training.
The amount of rewards he gets in each game is very low and struggles to increase.

## Improvements

- [ ] Implement a more effective rewards system
- [ ] Validate the implementation of Deep Q-Learning
- [ ] Identify an evaluation metric for the agent other than success rate or rewards obtained

## Goals

L'obiettivo principale è riuscire a creare un agente in grado di risolvere sudoku almeno di difficoltà media.

## Contribute

If you are interested in contributing to this project, you are welcome! Follow the steps below:

1. Fork this repository to your GitHub account.
2. Clone the forked repository to your local machine.
3. Create a new branch for your enhancements: `git checkout -b enhancements`.
4. Work on improvements and make your own changes in the code.
5. Make sure the code works correctly and is well documented.
6. Add and commit your changes: `git commit -am 'Added improvements: short description'`.
7. Push your branch to your GitHub repository: `git push origin improvements`.
8. Go to your forked repository on GitHub and open a new Pull Request with your changes.
9. Clearly describe the changes you made and the reasoning behind them.

Once the pull request is opened, it will be reviewed and, if deemed appropriate, integrated into the main project.

We thank you in advance for your contribution!

## License

This project is released under the [GNU General Public License v3.0 (GPL-3.0)](LICENSE).

The GPL-3.0 license is an open source license that grants users the freedom to use, modify and distribute the software. Some key features of the license include:

- **Freedom to use**: You can use the software for any purpose, commercial or non-commercial.
- **Freedom to modify**: You can modify the software as you wish and distribute your changes.
- **Distribution of modifications**: If you distribute the software or its modifications, you must make the source code and license rights available to end users.
- **No additional restrictions**: You cannot add additional restrictions to GPL-3.0 licenses.

For further details and information about the rights and obligations granted by the GPL-3.0 license, read the [LICENSE.txt] file in this repository.

### Credits

This project was created by Cristian Cerasuolo basing on the work of [PatrickLoeber](https://github.com/patrickloeber/snake-ai-pytorch) for the code workflow and on the work of [Kistler21](https://github.com/Kistler21) for the Sudoku interface.
