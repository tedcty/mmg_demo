# MAS_103 Muscle Attachment Mapping

This document formalizes the mapping of the `MAS_103` muscle attachment point clouds to their corresponding anatomical muscle names. The point clouds represent the areas where muscles originate or insert on the bones.

## Known Mappings

*   **Scapula**
    *   **ID 69**: Subscapularis (Attachment site on the subscapular fossa)

## Clavicle Attachments
The clavicle dataset provides explicit names rather than numerical IDs. Both left and right sides contain the following attachment point clouds:
*   `Deltoid_Muscle`
*   `Pectoralismajor_Muscle`
*   `Sternocleidomastoid_Muscle`
*   `Sternohyoid_Muscle`
*   `Subclavius_Muscle`
*   `Trapezius_Muscle`

## Unmapped Numerical IDs

The following numerical IDs exist as `.csv` and `.ply` files in the `MAS_103` dataset but currently lack a formalized mapping to an anatomical muscle name in this project. 

> [!IMPORTANT]
> Please update this table with the correct muscle names corresponding to the MAS_103 dataset nomenclature.

### Scapula
*   **47**, **49**, **51**
*   **67**, **68**, **70**, **71**, **72**
*   **73a**, **73b**, **74**, **76a**
*   **149**, **151**, **152**, **153**

### Humerus
*   **48**, **67**, **69**, **70**, **71**, **72**
*   **74**, **75**, **76b**, **76c**, **78a**, **79_80**
*   **81**, **82a**, **86**, **87**, **88**, **89**
*   **90**, **91**, **92**, **150**

### Radius
*   **73**, **78**, **82b**, **84**, **85**, **86**
*   **91**, **92**, **93**

### Ulna
*   **75**, **76**, **77**, **78b**, **81b**, **82a**
*   **83**, **85**, **90b**, **91**, **92**, **94**, **95**

## Data Source
All muscle attachment point clouds are located in:
`predict_gui/Resources/MAS_103/`

Each muscle site has two corresponding files (e.g., `69_NodeNo_2.csv` for vertices and `69_NodeNo_2.ply` for mesh representation).
