class Solution {
public:
    int bestClosingTime(string customers) {
        int n =customers.size();
        vector<int> shop(n+1);
        int nc=0;
        int yc=0;

        for(int i=0;i<n;i++){
           
            if(customers[i]=='N'){
                
                shop[i]=nc;
                nc++;
            }
            else shop[i]=nc;
            
             

        }
        
        for(int i=n-1;i>=0;i--){
            
            if(customers[i]=='Y')yc++;
            shop[i]+=yc;

        }
        shop[n]=nc;
        int mini=*min_element(shop.begin(),shop.end());
        

        for(int i=0;i<=n;i++){
            if(shop[i]==mini) return i;
        }
        return 0;
    }
};