using namespace std;

struct ClusterPair;

struct Cluster{
    int n,index,id;
    vector<float> ids;
    vector<vector<float>> elem;
    vector<ClusterPair*> pairs;
    vector<float> average;
    Cluster(vector<float> e ,int _id):n(1),index(_id),id(_id){
        this->ids.push_back(id);
        elem.push_back(e);
        this->average = e;
    }

    
};

struct ClusterPair{
    Cluster *a,*b;
    int index;
    int method;
    float dist;
    bool active;
    ClusterPair(Cluster* x,Cluster* y,int i,int _method):index(i){
        this->dist = 0;
        this->method = _method;
        this->a = x;
        this->b = y;
        this->recalc();
        this->active = true;
    }

    Cluster* getCluster(int i){
        if(i == 0)return this->a;
        else return this->b;
    }
    bool merge(int state){
        this->a->elem.insert(this->a->elem.end(),this->b->elem.begin(),this->b->elem.end());
        this->a->ids.insert(this->a->ids.end(),this->b->ids.begin(),this->b->ids.end());
        for(int i = 0;i < this->a->average.size();++i){
            this->a->average[i] = (this->a->average[i]*this->a->n + this->b->average[i]*this->b->n)/(float)(this->a->n + this->b->n);
        }
        this->a->n += this->b->n;
        this->a->index = state;
        return true;
    }

    bool recalc(void){
        float tmp = this->dist;
        this->dist = this->distance();
        return this->dist > tmp;
    }

    float distance(void){
        float ret = 1;
        if(this->method == 0){
            ret = 0;
            for(int i = 0;i < this->a->n;++i){
                for(int j = 0;j < this->b->n;++j){
                    float cnt = 0;
                    for(int k = 0;k < this->a->elem[0].size();++k){
                        cnt += (a->elem[i][k]-b->elem[j][k])*(a->elem[i][k]-b->elem[j][k]);
                    }
                    cnt = sqrt(cnt);
                    ret = max(ret,cnt);
                }
            }
        }else if(this->method == 1){
            ret = 1e10;
            for(int i = 0;i < this->a->n;++i){
                for(int j = 0;j < this->b->n;++j){
                    float cnt = 0;
                    for(int k = 0;k < this->a->elem[0].size();++k){
                        cnt += (a->elem[i][k]-b->elem[j][k])*(a->elem[i][k]-b->elem[j][k]);
                    }
                    cnt = sqrt(cnt);
                    ret = min(ret,cnt);
                }
            }
        }else if(this->method == 2){
            ret = 0;
            for(int i = 0;i < this->a->n;++i){
                for(int j = 0;j < this->b->n;++j){
                    float cnt = 0;
                    for(int k = 0;k < this->a->elem[0].size();++k){
                        cnt += (a->elem[i][k]-b->elem[j][k])*(a->elem[i][k]-b->elem[j][k]);
                    }
                    cnt = sqrt(cnt);
                    ret += cnt;
                }
            }
            ret /= (float)(this->a->n*this->b->n);
        }else if(this->method == 3){
            ret = 0;
            int sz = this->a->average.size();
            for(int i = 0; i < sz; ++i){
                ret += (this->a->average[i]-this->b->average[i])*(this->a->average[i]-this->b->average[i]);
            }
            ret *= (this->a->n*this->b->n)/(float)(this->a->n + this->b->n);
            ret = sqrt(ret);
        }
        return ret;
    }
};

class Heap{
    private:
        int d_size;
        int d_num;
       

    public:
        int n;
        int state;
        int size;
        vector<vector<float>> debug;
        vector<float> debug2;
        vector<float>k;
        float debug3;
        vector<ClusterPair*> heap;
        Heap(vector<vector<float>> input,int method){
            int sz = input.size();
            this->size = sz;
            this->state = sz;
            this->d_size = input[0].size();
            this->d_num = sz;
            this->n = sz*(sz-1)/2;
            this->heap.resize(this->n+1);
            vector<Cluster*> table;
            for(int i = 0; i < sz; ++i){
                Cluster* c = new Cluster(input[i],i);
                table.push_back(c);
            }
            int index = 1;
            for(int i = 0;i < sz-1;++i){
                for(int j = i+1;j < sz;++j){
                    ClusterPair* p = new ClusterPair(table[i],table[j],index,method);
                    this->heap[index] = p;
                    table[i]->pairs.push_back(p);
                    table[j]->pairs.push_back(p);
                    ++index;
                }
            }
            int x = this->n/2;
            while(x > 0){
                this->shiftDown(x);
                --x;
            }

        }

        void shiftUp(int start){
            int x = start;
            int next = x/2;
            while(next >= 1){
                float h_next = this->heap[next]->dist,h_now = this->heap[x]->dist;
                if(h_next < h_now)break;
                swap(this->heap[next]->index,this->heap[x]->index);
                swap(this->heap[next],this->heap[x]);
                x = next;
                next /= 2;
            }
        }

        void shiftDown(int start){

            int x = start;
            int next = 2*x;
            while(next <= this->n){
                if(next < this->n){
                    if(this->heap[next]->dist > this->heap[next+1]->dist)next++;
                }
                if(this->heap[next]->dist > this->heap[x]->dist)break;
                swap(this->heap[next]->index,this->heap[x]->index);
                swap(this->heap[next],this->heap[x]);
                x = next;
                next *= 2;

            }

        }

        void remove(int index){
            swap(this->heap[index]->index,this->heap[this->n]->index);
            swap(this->heap[index],this->heap[this->n]);
            --this->n;
            this->heap.erase(this->heap.end()-1);
            if(this->n > 1)this->shiftDown(index);
        }

        vector<float> update(void){
            float dist,cnt,num_a,num_b;
            auto pr = this->heap[1];
            dist = pr->dist; 
            num_a = pr->a->index;
            num_b = pr->b->index;
            pr->merge(this->state);

            debug = pr->a->elem;
            cnt = pr->a->n;
            for(int i =0 ;i < 2;i++){
                auto c = pr->getCluster(i);
                vector<int>tmp;
                for(int j = 0;j < c->pairs.size();++j){
                    if(c->pairs[j]->index == pr->index || !c->pairs[j]->active){
                        tmp.push_back(j);
                        continue;
                    }
                    if(i == 0){
                        bool flag = c->pairs[j]->recalc();
                        if(flag)this->shiftDown(c->pairs[j]->index);
                        else this->shiftUp(c->pairs[j]->index);  
                    }else{
                        this->remove(c->pairs[j]->index);
                        c->pairs[j]->active = false;
                    }
                }
                sort(tmp.begin(),tmp.end());
                for(int j = tmp.size()-1;j >= 0;--j){
                    c->pairs.erase(c->pairs.begin() + tmp[j]);
                }
            }
            this->remove(pr->index);
            vector<float> ret = {num_a,num_b,dist,cnt};
            this->state++;
            this->size--;
            return ret;
        }



        int judge(){
            int ret = -1;
            for(int i = this->n;i > 1;--i){
                int j = i/2;
                if(this->heap[j]->dist > this->heap[i]->dist){
                    ret = i;
                }
            }
            return ret;
        }

        vector<int> getClassify(void){
            vector<int>ret(this->d_num);
            set<int>indexs;
            int classify = 0;
            for(int i = 1; i <= this->n; ++i){
                for(int j = 0;j < 2; ++j){
                    auto c = this->heap[i]->getCluster(j);
                    if(indexs.count(c->id))continue;
                    for(int k = 0;k < c->ids.size();++k){
                        ret[c->ids[k]] = classify;
                    }
                    classify++;
                    indexs.insert(c->id);
                }
            }
            return ret;
        }
};