#include<bits/stdc++.h>
#include<fstream>

using namespace std;

int main(int argc, char *argv[])
{
	vector <unsigned int> v;
	srand(time(NULL));
	long long int i,num;
	num = (long long int)atoll(argv[1]);
	for(i=0;i<num;i++)
	{
		v.push_back(rand() % 65536);
	}
	sort(v.begin(),v.end(),greater<unsigned int>());
	ofstream fout;
	fout.open("reverse_dataset.txt");
	for(i=0;i<num-1;i++)
	{
		fout<<v[i]<<endl;
	}
	fout<<v[i];
	fout.close();
	return 0;
}