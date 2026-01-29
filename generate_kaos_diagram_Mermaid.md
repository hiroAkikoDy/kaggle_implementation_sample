graph LR
    %% ノード定義
    ROOT
    G1[G1]
    G2[G2]
    G3[G3]
    G4[G4]
    
    G11[G11]
    G12[G12]
    G13[G13]
    G111[G111]
    G112[G112]
    
    G21[G21]
    G22[G22]
    G23[G23]
    
    G31[G31]
    G32[G32]
    G33[G33]
    
    G41[G41]
    G42[G42]
    G43[G43]

    %% 依存関係（エッジ）
    ROOT --> G1
    ROOT --> G2
    ROOT --> G3
    ROOT --> G4

    G1 --> G11
    G1 --> G12
    G1 --> G13
    G11 --> G111
    G11 --> G112

    G2 --> G21
    G2 --> G22
    G2 --> G23

    G3 --> G31
    G3 --> G32
    G3 --> G33

    G4 --> G41
    G4 --> G42
    G4 --> G43

    %% スタイル設定（元の図のカラーリングを再現）
    style ROOT fill:#FFB6C1,stroke:#333,stroke-width:2px
    style G1 fill:#ADD8E6,stroke:#333
    style G11 fill:#ADD8E6,stroke:#333
    style G111 fill:#ADD8E6,stroke:#333
    style G112 fill:#ADD8E6,stroke:#333
    style G12 fill:#ADD8E6,stroke:#333
    style G13 fill:#ADD8E6,stroke:#333
    
    style G2 fill:#90EE90,stroke:#333
    style G21 fill:#90EE90,stroke:#333
    style G22 fill:#90EE90,stroke:#333
    style G23 fill:#90EE90,stroke:#333
    
    style G3 fill:#FFFFE0,stroke:#333
    style G31 fill:#FFFFE0,stroke:#333
    style G32 fill:#FFFFE0,stroke:#333
    style G33 fill:#FFFFE0,stroke:#333
    
    style G4 fill:#E6E6FA,stroke:#333
    style G41 fill:#E6E6FA,stroke:#333
    style G42 fill:#E6E6FA,stroke:#333
    style G43 fill:#E6E6FA,stroke:#333