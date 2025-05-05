import cv2
from  ultralytics import YOLO
import os

#1- model=YOLO("yolov8s.pt")  #loading the yolo object model When you run this code for the first time, it will download the yolov8m.pt file from the Ultralytics server to the current folder. Then it will construct the model object
#model.train(data="./data/data.yaml", epochs=8)


#2- apres entraainement
model = YOLO("./runs/detect/train4/weights/best.pt")
results = model.predict(source="C:/Users/PC/Desktop/dataset/data/test/images", save=True, save_txt=True, project="./runs/detect", name="predict19")


#METHODE 1
chemin_label="./runs/detect/predict19/labels"

for i in range(len(results)):
    img = results[i]
    item_box = img.boxes.conf   #item_box contient les proba des box se trouvant dans une image TYPE TENSOR

         #si on a au moins 2 boxes
    chemin=img.path        #retourne le chemin de l iamge
    label_name=os.path.splitext(os.path.basename(chemin))[0]+".txt"  #img1 le nom de l image sans le xtension
    labels_path=os.path.join(chemin_label,label_name) #on obtenu le nom du path

            #=======================================
    if os.path.exists(labels_path):
        with open(labels_path,"r") as file:
                    lignes = []
                    k = file.readlines()
                    for j in k: #deux lignes de matrcies en cas de deux bounding boxes
                        lignes.append(j.strip().split())  #coupe par les espaces ['0', '0.412', '0.531', '0.221', '0.123']['0', '0.652', '0.732', '0.175', '0.085']

                #=======================================
        if len(lignes) < 2:
            print(f"⚠!!! Less than 2 boxes for {chemin}")
            continue
        if len(lignes) >= 2:
            boxes_left=[]     #cette liste contient les indices des boxes qu on va garder => c est par c est indice qu on peut acceder a les lignes de la matrice pour recuperer les w,l
            #Step 1
            for prob in range(len(item_box)):     #prob presente l indice
                    if item_box[prob].item()>=0.7:
                        boxes_left.append(lignes[prob])
                        #list.append(prob)
            #Step2
            boxes_left1 = []
            for h in range(len(boxes_left)):
                box = boxes_left[h]
                if float(box[3])>float(box[4]):     #convertie de chaine de caractere en nbre pour comp
                           boxes_left1.append(box)

            #Step3
            if len(boxes_left1)>0:
                    best_box = None   #juste pour traiter le cas ou toute on verifier la premiere condition
                    air_max=0

                    for s in range(len(boxes_left1)):
                        air=float(boxes_left1[s][3]) * float(boxes_left1[s][4])   #car ce st en chaine car
                        if air > air_max:
                            air_max = air
                            best_box = boxes_left1[s]
                    if best_box:
                        with open(labels_path, "w") as f:
                            f.write(" ".join(best_box) + "\n")
            else:
                    os.remove(labels_path)

        else:
            print(f"Moins de 2 boxes pour {img.path}")
            continue  # Passe aussi à l'image suivante si < 2 boxes












#largeur = float(parts[3])
#hauteur = float(parts[4])
