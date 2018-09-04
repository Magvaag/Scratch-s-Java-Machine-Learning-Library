import net.vaagen.ActivationFunction;
import net.vaagen.FeedForwardNeuralNetwork;
import net.vaagen.FeedForwardStateMachine;
import net.vaagen.utility.NeuralUtility;

import java.util.Random;

/**
 * Created by Magnus on 8/31/2018.
 */
public class ThreeInARow {

    public static void main(String[] args) {
        new ThreeInARow();
    }

    public ThreeInARow() {
        // INPUT: The values of the board, 0 for Empty, 1 for Player1, -1 for Player2
        // OUTPUT: The estimated reward for each position

        // REWARD: Feed the network 1 if game won, -1 if game lost, and 0 if game ongoing.

        Random random = new Random();
        int[] networkStructure = new int[] {9, 18, 9};
        float learningRate = .1f;

        float greedyStart = 1f;
        float greedyChange = .997f;
        float greedyMin = .01f;



        FeedForwardNeuralNetwork player = new FeedForwardNeuralNetwork(networkStructure, learningRate, ActivationFunction.SIGMOID);
        FeedForwardNeuralNetwork dummy = new FeedForwardNeuralNetwork(networkStructure, learningRate, ActivationFunction.SIGMOID);

        FeedForwardStateMachine stateMachine = new FeedForwardStateMachine();
        Game game = new Game();
        game.setRandomStartingPlayer();

        // Loop through until the game is over
        while (game.getGameState() == Game.STATE_ONGOING) {
            if (game.nextPlayer == Game.PLAYER_1) {
                // TODO : Implement greedy
                int[] boardCopy = game.board.clone();
                int decision = neuralNetworkMakeMove(player, game);

                // Save the state so we can change the weights later
                stateMachine.saveState(NeuralUtility.intArrayToFloat(boardCopy), decision);
            } else {
                neuralNetworkMakeMove(dummy, game);
            }
        }

        game.print();
    }

    public int neuralNetworkMakeMove(FeedForwardNeuralNetwork ffnn, Game game) {
        float[] output = ffnn.calc(NeuralUtility.intArrayToFloat(game.board));
        int[] sortedOutput = NeuralUtility.sortOutput(output);

        // Place the piece. If the slot is taken, try the second best piece
        for (int i = 0; i < sortedOutput.length; i++) {
            if (game.place(sortedOutput[i])) {
                return sortedOutput[i];
            }
        }

        return -1;
    }

    public static class Game {

        public static int PLAYER_1 = 1, PLAYER_2 = -1, EMPTY = 0;
        public static int STATE_ONGOING = 0, STATE_PLAYER_1 = PLAYER_1, STATE_PLAYER_2 = PLAYER_2, STATE_DRAW = 2;

        // 0, 1, 2
        // 3, 4, 5
        // 6, 7, 8
        private int[] board;
        private int nextPlayer;
        private int piecesPlaced;

        public Game() {
            board = new int[9];
            piecesPlaced = 0;
            nextPlayer = PLAYER_1;
        }

        public void setRandomStartingPlayer() {
            nextPlayer = new Random().nextBoolean() ? PLAYER_1 : PLAYER_2;
        }

        public boolean place(int position) {
            if (board[position] != EMPTY) return false;
            board[position] = nextPlayer;
            nextPlayer = nextPlayer == PLAYER_1 ? PLAYER_2 : PLAYER_1;
            piecesPlaced++;
            return true;
        }

        // Return 0 for
        public int getGameState() {
            if (piecesPlaced == 9) return STATE_DRAW;

            // Check vertical
            for (int y = 0; y < 3; y++) {
                int val = board[0 + 3*y] + board[1 + 3*y] + board[2 + 3*y];
                if (val == PLAYER_1 * 3) return STATE_PLAYER_1;
                if (val == PLAYER_2 * 3) return STATE_PLAYER_2;
            }

            // Check horizontal
            for (int x = 0; x < 3; x++) {
                int val = board[0 + x] + board[3 + x] + board[6 + x];
                if (val == PLAYER_1 * 3) return STATE_PLAYER_1;
                if (val == PLAYER_2 * 3) return STATE_PLAYER_2;
            }

            // Check cross
            int val = board[0] + board[4] + board[8];
            if (val == PLAYER_1 * 3) return STATE_PLAYER_1;
            if (val == PLAYER_2 * 3) return STATE_PLAYER_2;
            val = board[2] + board[4] + board[6];
            if (val == PLAYER_1 * 3) return STATE_PLAYER_1;
            if (val == PLAYER_2 * 3) return STATE_PLAYER_2;

            // Game still going
            return STATE_ONGOING;
        }

        public void print() {
            System.out.println("=======");
            System.out.println(board[0] + ", " + board[1] + ", " + board[2]);
            System.out.println(board[3] + ", " + board[4] + ", " + board[5]);
            System.out.println(board[6] + ", " + board[7] + ", " + board[8]);
            System.out.println("State: " + getGameState());
            System.out.println("=======");
        }

    }

}
