//TODO UDP SEND TO 127.0.0.1:4242 OF POSITION X Y Z (0,8,16) ROTATION X Y Z (24,32,40)//




#include <Winsock2.h>
#include <Ws2tcpip.h>


#include <iostream>
#include <string>


class UDPSender
{
private:
    const int BUFFER_SIZE = sizeof(double) * 6;
    double position_data[6];

    sockaddr_in dest;
    sockaddr_in local;

    WSAData data;
    SOCKET s;

public:
    std::string ip;
    int port;

    UDPSender(const char* dest_ip, int dest_port);
    //~UDPSender();

    /**
        Sends a data vector to opentrack.
        @param data: Size 6 array which contains [X,Y,Z,Yaw,Pitch,Roll].
    */
    void send_data(double* data);
    void NormalizeBetweenTwoNumber(int min, int max, double* d, int size);
};



UDPSender::UDPSender(const char* dest_ip, int dest_port)
{

    this->ip = std::string(dest_ip);
    this->port = dest_port;

    //std::cout << "ip: " << this->ip << "  port: " <<  this->port << std::endl;

    dest = sockaddr_in();
    local = sockaddr_in();

    WSAStartup(MAKEWORD(2, 2), &data);

    local.sin_family = AF_INET;
    inet_pton(AF_INET, dest_ip, &local.sin_addr.s_addr);
    local.sin_port = htons(0);

    dest.sin_family = AF_INET;
    inet_pton(AF_INET, dest_ip, &dest.sin_addr.s_addr);
    dest.sin_port = htons(dest_port);

    s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    bind(s, (sockaddr*)&local, sizeof(local));
}

//TODO: find why this closes connection every frame and fix
// 
//UDPSender::~UDPSender()
//{
//    std::cout << "Closing connection" << std::endl;
//    closesocket(s);
//    WSACleanup();
//}



void UDPSender::send_data(double* d)
{

    double* tmp = new double[6];
    double* coords = new double[3];
    coords[0] = d[0] / 20;
    coords[1] = d[1] / 20;
    coords[2] = d[2] * 100;
    double* angles = new double[3];
    angles[0] = d[3];
    angles[1] = d[4];
    angles[2] = d[5];
    NormalizeBetweenTwoNumber(-35, 35, angles, 3);

    tmp[0] = - coords[0];
    tmp[1] = - coords[1];
    tmp[2] = coords[2];
    tmp[3] =  - angles[2];
    tmp[4] =  - angles[1];
    tmp[5] =   - angles[0];
    
    //printf("X: %lf Y: %lf Z: %lf Yaw: %lf Pitch: %lf Roll: %lf\n", d[0], d[1], d[2], d[3], d[4], d[5]);

    // Make packet
    const char* pkt = (char*)tmp;
    sendto(s, pkt, BUFFER_SIZE, 0, (sockaddr*)&dest, sizeof(dest));
};

void UDPSender::NormalizeBetweenTwoNumber(int min, int max, double* d, int size)
{
    double* tmp = new double[size];
    for (int i = 0; i < size; i++) {
        tmp[i] = ((max - min) * ((d[i] - (-1)) / (1 - (-1))) + min);
        d[i] = tmp[i];
    }
}


