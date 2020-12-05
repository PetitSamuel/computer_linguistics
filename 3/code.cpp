#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

// want to represents vocab items by integers because then various tables
// need by the IBM model and EM training can just be represented as 2-dim
// tables indexed by integers

// the following #defines, defs of VS, VO, S, O, and create_vocab_and_data()
// are set up to deal with the specific case of the two pair corpus
// (la maison/the house)
// (la fleur/the flower)

// S VOCAB
#define LA 0
#define MAISON 1
#define FLEUR 2
// O VOCAB
#define THE 0
#define HOUSE 1
#define FLOWER 2

#define VS_SIZE 3
#define VO_SIZE 3
#define D_SIZE 2

vector<string> VS(VS_SIZE); // S vocab: VS[x] gives Src word coded by x
vector<string> VO(VO_SIZE); // O vocab: VO[x] gives Obs word coded by x

vector<vector<int>> S(D_SIZE); // all S sequences; in this case 2
vector<vector<int>> O(D_SIZE); // all O sequences; in this case 2

// sets S[0] and S[1] to be the int vecs representing the S sequences
// sets O[0] and O[1] to be the int vecs representing the O sequences
void create_vocab_and_data();

// functions which use VS and VO to 'decode' the int vecs representing the
// Src and Obs sequences
void show_pair(int d);
void show_O(int d);
void show_S(int d);

// Amount of iterations to use - set to 50 to mimick sample output
#define COUNT 50

double proba[VO_SIZE][VS_SIZE];
double unnormalised_count[VO_SIZE][VS_SIZE];

int main()
{
    create_vocab_and_data();

    // initialise uniformly (1/3)
    for (int i = 0; i < VO_SIZE; i++)
        for (int j = 0; j < VS_SIZE; j++)
            proba[i][j] = 1.0 / 3.0;

    for (int c = 0; c < COUNT; c++)
    {
        // initialise exponent counts to 0
        for (int i = 0; i < VO_SIZE; i++)
            for (int j = 0; j < VS_SIZE; j++)
                unnormalised_count[i][j] = 0;

        std::vector<int> obs_vect;
        std::vector<int> src_vect;
        for (int i = 0; i < D_SIZE; i++)
        {
            obs_vect = O[i];
            src_vect = S[i];
            for (int j = 0; j < obs_vect.size(); j++)
            {
                int obs_word = obs_vect[j];
                double obs_word_probability = 0.0;
                for (int k = 0; k < src_vect.size(); k++)
                {
                    int src_word = src_vect[k];
                    obs_word_probability += proba[obs_word][src_word];
                }
                for (int l = 0; l < src_vect.size(); l++)
                {
                    int src_word = src_vect[l];
                    if (proba[obs_word][src_word] > 0)
                    {
                        unnormalised_count[obs_word][src_word] += proba[obs_word][src_word] / obs_word_probability;
                    }
                }
            }
        }

        for (int s = 0; s < VS_SIZE; s++)
        {
            double normalise = 0.0;
            for (int o = 0; o < VO_SIZE; o++)
            {
                normalise += unnormalised_count[o][s];
            }
            if (normalise > 0)
                for (int o = 0; o < VO_SIZE; o++)
                    proba[o][s] = unnormalised_count[o][s] / normalise;
            else
                for (int o = 0; o < VO_SIZE; o++)
                    proba[o][s] = normalise;
        }

        cout << endl
             << "unnormalised counts in iteration " << c << endl;
        for (int y = 0; y < VS_SIZE; y++)
            for (int x = 0; x < VO_SIZE; x++)
                cout << VO[x] << "   " << VS[y] << "   " << unnormalised_count[x][y] << endl;

        cout << endl
             << "after iteration " << c << " trans probs (tr(o|s)):" << endl;
        for (int x = 0; x < VO_SIZE; x++)
            for (int y = 0; y < VS_SIZE; y++)
                cout << VO[x] << "   " << VS[y] << "   " << proba[x][y] << "\n";
    }
}

void create_vocab_and_data()
{

    VS[LA] = "la";
    VS[MAISON] = "maison";
    VS[FLEUR] = "fleur";

    VO[THE] = "the";
    VO[HOUSE] = "house";
    VO[FLOWER] = "flower";

    cout << "source vocab\n";
    for (int vi = 0; vi < VS_SIZE; vi++)
    {
        cout << VS[vi] << " ";
    }
    cout << endl;
    cout << "observed vocab\n";
    for (int vj = 0; vj < VO_SIZE; vj++)
    {
        cout << VO[vj] << " ";
    }
    cout << endl;

    // make S[0] be {LA,MAISON}
    //      O[0] be {THE,HOUSE}
    S[0].resize(2);
    O[0].resize(2);
    S[0] = {LA, MAISON};
    O[0] = {THE, HOUSE};

    // make S[1] be {LA,FLEUR}
    //      O[1] be {THE,FLOWER}
    S[1].resize(2);
    O[1].resize(2);
    S[1] = {LA, FLEUR};
    O[1] = {THE, FLOWER};

    for (int d = 0; d < S.size(); d++)
    {
        show_pair(d);
    }
}

void show_O(int d)
{
    for (int i = 0; i < O[d].size(); i++)
    {
        cout << VO[O[d][i]] << " ";
    }
}

void show_S(int d)
{
    for (int i = 0; i < S[d].size(); i++)
    {
        cout << VS[S[d][i]] << " ";
    }
}

void show_pair(int d)
{
    cout << "S" << d << ": ";
    show_S(d);
    cout << endl;
    cout << "O" << d << ": ";
    show_O(d);
    cout << endl;
}
