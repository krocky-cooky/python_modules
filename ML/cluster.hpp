using namespace std;

struct Cluster{
    int n,id;
    vector<vector<float>> elem;
    Cluster(vector<float> e,int _id):n(1),id(_id){
        elem.push_back(e);
    }
};

class Heap{
    private:
        int d_size;
        map<pair<int,int>,int>where;
        vector<pair<int,int>>pr;
        vector<Cluster> table;

    public:
        int n;
        int flag;
        vector<float> heap;
        Heap(vector<vector<float>> input){
            heap.push_back(0);
            pr.push_back({0,0});
            int sz = input.size();
            this->d_size = input[0].size();
            this->n = sz*(sz-1)/2;
            for(int i = 0; i < sz; ++i){
                Cluster c(input[i],i);
                this->table.push_back(c);
            }
            int index = 0;
            for(int i = 0;i < sz-1;++i){
                for(int j = i+1;j < sz;++j){
                    this->pr.push_back({i,j});
                    this->heap.push_back(this->distance(this->table[i],this->table[j]));
                    this->where[make_pair(i,j)] = index;
                    ++index;
                }
            }
            int x = this->n/2;
            while(x > 0){
                this->shiftDown(x);
                --x;
            }
            this->flag = this->judge();
    

        }

        void shiftUp(int start){
            int x = start;
            int next = x/2;
            while(x > 0){
                int h_next = this->heap[next],h_now = this->heap[x];
                if(h_next < h_now)break;
                swap(this->heap[next],this->heap[x]);
                swap(this->pr[next],this->pr[x]);
                swap(this->where[this->pr[next]],this->where[this->pr[x]]);
                x = next;
                next /= 2;
            }
        }

        void shiftDown(int start){
            int x = start;
            int next = 2*x;
            while(next <= this->n){
                if(this->heap[next] > this->heap[next+1])next++;
                if(this->heap[next] > this->heap[x])break;
                swap(this->heap[next],this->heap[x]);
                swap(this->pr[next],this->pr[x]);
                swap(this->where[this->pr[next]],this->where[this->pr[x]]);
                x = next;
                next *= 2;

            }
        }

        void remove(int index){
            swap(this->heap[index],this->heap[this->n]);
            swap(this->pr[index],this->pr[this->n]);
            swap(this->where[this->pr[index]],this->where[this->pr[this->n]]);
            --this->n;
            this->heap.erase(this->heap.end()-1);
            this->pr.erase(this->pr.end()-1);
            this->shiftDown(index);
        }

        float distance(Cluster a,Cluster b){
            float ret = 0;
            for(int i = 0;i < a.n;++i){
                for(int j = 0;j < b.n;++j){
                    float cnt = 0;
                    for(int k = 0;k < this->d_size;++k){
                        cnt += (a.elem[i][k]-b.elem[j][k])*(a.elem[i][k]-b.elem[j][k]);
                    }
                    cnt = sqrt(cnt);
                    ret = max(ret,cnt);
                }
            }
            return ret;
        }

        int judge(){
            int ret = -1;
            for(int i = this->n;i > 1;--i){
                int j = i/2;
                if(this->heap[j] > this->heap[i]){
                    ret = i;
                }
            }
            return ret;
        }

        float calc(vector<int> a,vector<int> b){
            int s = a.size();
            float ans = 0;
            for(int i = 0;i < s; i++){
                ans += a[i]*b[i];
            }
            return ans;
        }

};